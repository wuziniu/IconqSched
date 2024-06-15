import numpy as np
import os
import logging
import pandas as pd
from typing import Optional, Tuple
from models.single.stage import SingleStage
from models.concurrency.complex_models import ConcurrentRNN
from utils.load_brad_trace import create_concurrency_dataset


def pre_process_snowset(df):
    df["createdTime"] = pd.to_datetime(df["createdTime"], format="mixed")
    df = df.sort_values(by=["createdTime"], ascending=True)
    df["timestamp_s"] = df["createdTime"].astype("int64") / 1e9
    this_min = df["timestamp_s"].min()
    df["g_offset_since_start_s"] = df["timestamp_s"] - this_min
    df["run_time_s"] = df["durationTotal"] / 1000
    return df


class TraceManager:
    """
    This class generates a real-world workload trace (e.g. snowset).
    It replays the workload trace on a real-world IMDB dataset and corresponding queries.
    It makes sure the queries are submitted at the provide timestamp and
         try to make sure each query has the same runtime as in the original trace.
    """

    def __init__(
        self,
        stage_model: SingleStage,
        predictor: ConcurrentRNN,
        query_file: str,
        database: str,
        debug: bool = False,
        logger: Optional[logging.Logger] = None,
        stride: int = 4,
        timeout: int = 400
    ):
        self.stage_model = stage_model
        self.predictor = predictor
        self.database = database
        if database == "redshift":
            os.environ["PGCLIENTENCODING"] = "utf-8"
        self.debug = debug
        self.logger = logger

        self.stride = stride
        self.timeout = timeout

        with open(query_file, "r") as f:
            self.query_sql = f.readlines()
        self.average = np.zeros(len(self.query_sql))
        self.std = np.zeros(len(self.query_sql))
        self.prepare()

    def prepare(self):
        for i in range(len(self.query_sql)):
            self.average[i] = self.stage_model.cache.running_average[i]
            self.std[i] = self.stage_model.cache.running_std[i]
        self.short_running_queries = np.where(self.average < 5)[0]
        self.avg_runtime_interval = np.asarray([5, 30, 60, 100, np.infty])
        self.avg_runtime_idx_by_interval = []
        for i, low in enumerate(self.avg_runtime_interval[:-1]):
            high = self.avg_runtime_interval[i + 1]
            idx = np.where((self.average < high) & (self.average > low))[0]
            self.avg_runtime_idx_by_interval.append(idx)

    def simulate_trace_one_iter(self, trace: pd.DataFrame) -> Tuple[pd.DataFrame, bool]:
        concurrency_df = create_concurrency_dataset(
            trace, engine=None, pre_exec_interval=200
        )
        concurrency_df = concurrency_df.sort_values(by=["start_time"], ascending=True)
        concurrency_df["runtime"] = concurrency_df["ground_truth_runtime"]
        all_prediction, _ = self.predictor.predict(
            concurrency_df, return_per_query=False
        )
        ground_truth_runtime = np.minimum(concurrency_df["ground_truth_runtime"].values, self.timeout)
        all_new_runtime = []
        all_query_idx = []
        original_query_idx = concurrency_df["query_idx"].values
        query_rt_sort_idx = np.argsort(self.average)
        converge = False
        modified = 0
        modified_plus = 0
        modified_minus = 0
        for i in range(len(concurrency_df)):
            if ground_truth_runtime[i] <= 5:
                query_idx = int(np.random.choice(self.short_running_queries))
                runtime = self.average[query_idx]
            else:
                idx_sorted = int(
                    np.where(query_rt_sort_idx == original_query_idx[i])[0]
                )
                interval = (
                    np.searchsorted(self.avg_runtime_interval, all_prediction[i]) - 1
                )
                if interval == 0:
                    stride = int(np.abs(np.random.normal(self.stride, 2)) + 1)
                elif interval == 1:
                    stride = int(np.abs(np.random.normal(self.stride / 2, 2)) + 1)
                elif interval == 2:
                    stride = int(np.abs(np.random.normal(self.stride / 4, 1)) + 1)
                else:
                    stride = 1
                oqi = int(original_query_idx[i])
                if ground_truth_runtime[i] - all_prediction[i] > min(
                    max(10, self.std[oqi]), 50
                ):
                    modified += 1
                    modified_minus += 1
                    query_idx = query_rt_sort_idx[
                        min(idx_sorted + stride, len(query_rt_sort_idx) - 1)
                    ]
                    runtime = all_prediction[i] + np.abs(
                        np.random.normal(
                            all_prediction[i] / 4, np.sqrt(all_prediction[i])
                        )
                    )
                elif all_prediction[i] - ground_truth_runtime[i] > min(
                    max(10, self.std[oqi]), 50
                ):
                    modified += 1
                    modified_plus += 1
                    query_idx = query_rt_sort_idx[max(idx_sorted - stride, 0)]
                    runtime = all_prediction[i] - min(
                        np.abs(
                            np.random.normal(
                                all_prediction[i] / 4, np.sqrt(all_prediction[i])
                            )
                        ),
                        all_prediction[i] / 2,
                    )
                else:
                    query_idx = int(original_query_idx[i])
                    runtime = all_prediction[i]
            all_query_idx.append(query_idx)
            all_new_runtime.append(runtime)
        if self.debug:
            if self.logger is None:
                print(f"Modified {modified} queries out of {len(trace)} queries")
                print(
                    f"Modified plus {modified_plus} queries;  modified minus {modified_minus} queries"
                )
            else:
                self.logger.info(
                    f"Modified {modified} queries out of {len(trace)} queries"
                )
                self.logger.info(
                    f"Modified plus {modified_plus} queries;  modified minus {modified_minus} queries"
                )
        if modified / len(trace) < 0.01:
            converge = True
        new_df = pd.DataFrame(
            {
                "query_idx": np.asarray(all_query_idx),
                "ground_truth_runtime": ground_truth_runtime,
                "run_time_s": np.asarray(all_new_runtime),
                "g_offset_since_start_s": concurrency_df["start_time"].values,
            }
        )
        return new_df, converge

    def simulate_trace_iteratively(
        self, trace_path: str, max_iter: int = 10
    ) -> pd.DataFrame:
        trace = pd.read_csv(trace_path)
        trace = trace.sort_values(by=["g_offset_since_start_s"], ascending=True)
        all_runtime = trace["run_time_s"].values
        all_avg_runtime = []
        all_query_idx = []

        for i in range(len(trace)):
            if all_runtime[i] <= 5:
                query_idx = np.random.choice(self.short_running_queries)
            else:
                interval = (
                    np.searchsorted(self.avg_runtime_interval, all_runtime[i]) - 1
                )
                query_idx = np.random.choice(self.avg_runtime_idx_by_interval[interval])
            all_query_idx.append(query_idx)
            all_avg_runtime.append(self.average[query_idx])

        iter_df = pd.DataFrame(
            {
                "query_idx": np.asarray(all_query_idx),
                "ground_truth_runtime": all_runtime,
                "run_time_s": np.asarray(all_avg_runtime),
                "g_offset_since_start_s": trace["g_offset_since_start_s"].values,
            }
        )
        for iter in range(max_iter):
            if self.debug:
                if self.logger is None:
                    print(f"====================Iteration {iter} ====================")
                else:
                    self.logger.info(
                        f"====================Iteration {iter} ===================="
                    )
            iter_df, converge = self.simulate_trace_one_iter(iter_df)
            if converge:
                break

        iter_df["query_sql"] = [self.query_sql[i] for i in iter_df["query_idx"].values]
        return iter_df
