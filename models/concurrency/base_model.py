import numpy as np
import pandas as pd
from typing import Optional, List


class ConcurPredictor:
    def __init__(self):
        self.isolated_rt_cache = dict()
        self.average_rt_cache = dict()

    def get_isolated_runtime_cache(
        self,
        trace_df: pd.DataFrame,
        isolated_trace_df: Optional[pd.DataFrame] = None,
        get_avg_runtime: bool = False,
    ):
        if isolated_trace_df is None:
            isolated_trace_df = trace_df[trace_df["num_concurrent_queries"] == 0]
        # ignore queries that do not run insolation
        for i, rows in isolated_trace_df.groupby("query_idx"):
            if "runtime" in isolated_trace_df.columns:
                self.isolated_rt_cache[i] = np.median(rows["runtime"])
            else:
                self.isolated_rt_cache[i] = np.median(rows["run_time_s"])
        if get_avg_runtime:
            for i, rows in trace_df.groupby("query_idx"):
                if "runtime" in isolated_trace_df.columns:
                    self.average_rt_cache[i] = np.mean(rows["runtime"])
                else:
                    self.average_rt_cache[i] = np.mean(rows["run_time_s"])

    def train(self, trace_df: pd.DataFrame, isolated_trace_df: pd.DataFrame):
        raise NotImplemented

    def predict(self, eval_trace_df: pd.DataFrame, use_global: bool):
        raise NotImplemented

    def evaluate_performance(
        self,
        eval_trace_df: pd.DataFrame,
        use_global: bool = False,
        interval: List[float] = None,
    ):
        predictions, labels = self.predict(eval_trace_df, use_global)
        pred_all = []
        labels_all = []
        result_overall = []
        result_per_query = dict()
        pred_by_interval = dict()
        labels_by_interval = dict()
        result_by_interval = dict()
        for i in predictions:
            result_per_query[i] = []
            pred_all.append(predictions[i])
            labels_all.append(labels[i])
            abs_error = np.abs(predictions[i] - labels[i])
            q_error = np.maximum(predictions[i] / labels[i], labels[i] / predictions[i])
            for p in [50, 90, 95]:
                result_per_query[i].append(np.percentile(abs_error, p))
                result_per_query[i].append(np.percentile(q_error, p))
            if interval is not None:
                for j in range(len(interval)):
                    low = interval[j]
                    if j + 1 < len(interval):
                        high = interval[j + 1]
                    else:
                        high = np.infty
                    rt_i = np.median(labels[i])
                    if low <= rt_i < high:
                        if j not in pred_by_interval:
                            pred_by_interval[j] = []
                            labels_by_interval[j] = []
                            result_by_interval[j] = []
                        pred_by_interval[j].append(predictions[i])
                        labels_by_interval[j].append(labels[i])

        pred_all = np.concatenate(pred_all)
        labels_all = np.concatenate(labels_all)
        abs_error = np.abs(pred_all - labels_all)
        q_error = np.maximum(pred_all / labels_all, labels_all / pred_all)
        for p in [50, 90, 95]:
            p_a = np.percentile(abs_error, p)
            p_q = np.percentile(q_error, p)
            print(f"{p}% absolute error is {p_a}, q-error is {p_q}")
            result_overall.append(p_a)
            result_overall.append(p_q)
        if interval is not None:
            for j in range(len(interval)):
                low = interval[j]
                if j + 1 < len(interval):
                    high = interval[j + 1]
                else:
                    high = np.infty
                pred = np.concatenate(pred_by_interval[j])
                label = np.concatenate(labels_by_interval[j])
                print(
                    "================================================================"
                )
                print(
                    f"For query in range {low}s to {high}s, there are {len(label)} executions"
                )
                abs_error = np.abs(pred - label)
                q_error = np.maximum(pred / label, label / pred)
                for p in [50, 90, 95]:
                    p_a = np.percentile(abs_error, p)
                    p_q = np.percentile(q_error, p)
                    print(f"{p}% absolute error is {p_a}, q-error is {p_q}")
                    result_by_interval[j].append(p_a)
                    result_by_interval[j].append(p_q)
        return result_overall, result_per_query, result_by_interval
