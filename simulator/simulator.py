import numpy as np
from typing import Optional, Tuple
from utils.load_brad_trace import (
    load_trace,
    create_concurrency_dataset,
)
from scheduler.base_scheduler import BaseScheduler


class QueryBank:
    def __init__(
        self, sql_query_file: str, query_runtime_path: str, seed: int = 0
    ) -> None:
        with open(sql_query_file, "r") as f:
            sql_queries = f.readlines()
        query_runtime = np.load(query_runtime_path)
        assert len(sql_queries) == len(query_runtime)
        idx = np.argsort(query_runtime)
        self.query_runtime = query_runtime[idx]
        self.sql_queries = [sql_queries[i] for i in idx]
        self.query_len = len(self.query_runtime)
        np.random.seed(seed)

    def random_sample(self) -> (str, float):
        # make a random sample of the query
        idx = np.random.randint(self.query_len)
        return self.sql_queries[idx], self.query_runtime[idx]

    def sample_by_runtime(self, runtime: float) -> (str, float):
        # sample a query that best matches the runtime
        idx = np.searchsorted(self.query_runtime, runtime)
        idx = max(idx, self.query_len - 1)
        return self.sql_queries[idx], self.query_runtime[idx]


class Simulator:
    def __init__(
        self,
        scheduler: BaseScheduler,
        query_bank: Optional[QueryBank] = None,
        pause_wait_s: float = 5.0,
    ):
        self.scheduler = scheduler
        self.query_bank = query_bank
        self.pause_wait_s = pause_wait_s

    def replay_one_query(
        self,
        start_time: float,
        next_query_start_time: Optional[float] = None,
        query_str: Optional[int] = None,
        query_idx: Optional[int] = None,
    ):
        (
            should_immediate_re_ingest,
            should_pause_and_re_ingest,
            scheduled_submit,
        ) = self.scheduler.ingest_query(
            start_time, query_str=query_str, query_sql=None, query_idx=query_idx, simulation=True
        )
        if should_immediate_re_ingest:
            # the scheduler schedules one query at a time even if there are multiple queries in the queue,
            # so need to call again
            self.replay_one_query(start_time + 0.001, next_query_start_time)
        if should_pause_and_re_ingest:
            # this indicates it is not optimal to submit any query in the queue, will try in a future time
            if (
                next_query_start_time is not None
                and next_query_start_time <= start_time + self.pause_wait_s
            ):
                return
            self.replay_one_query(start_time + self.pause_wait_s, next_query_start_time)

    def finish_all_queries(self, last_timestamp: float):
        start_t = last_timestamp
        while len(self.scheduler.queued_queries) != 0:
            # make sure all queries are submitted
            self.replay_one_query(start_t + self.pause_wait_s, None)
            start_t += self.pause_wait_s
        # finish executing all submitted queries
        self.scheduler.finish_query_simulation(np.infty)

    def replay_workload(self, directory: str) -> Tuple[np.ndarray, np.ndarray]:
        all_raw_trace, all_trace = load_trace(directory, 8, concat=True)
        concurrency_df = create_concurrency_dataset(
            all_trace, engine=None, pre_exec_interval=200
        )
        concurrency_df = concurrency_df.sort_values(by=["start_time"], ascending=True)
        original_predictions = self.scheduler.make_original_prediction(concurrency_df)
        assert len(concurrency_df) == len(original_predictions)
        original_runtime = []
        all_start_time = concurrency_df["start_time"].values
        all_query_idx = concurrency_df["query_idx"].values
        for i in range(len(concurrency_df)):
            original_runtime.append(original_predictions[i])
            # replaying the query one-by-one
            if i < len(concurrency_df) - 1:
                next_query_start_time = all_start_time[i + 1]
            else:
                next_query_start_time = None
            self.replay_one_query(
                all_start_time[i], next_query_start_time, i, all_query_idx[i]
            )
        # finish all queries
        self.finish_all_queries(all_start_time[-1])
        new_runtime = []
        for i in range(len(concurrency_df)):
            new_runtime.append(self.scheduler.all_query_runtime[i])
        original_runtime = np.asarray(original_runtime)
        new_runtime = np.asarray(new_runtime)
        return original_runtime, new_runtime
