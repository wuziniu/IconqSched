import psycopg
import asyncio
import time
import numpy as np
from typing import Optional, Tuple, Union, MutableMapping, List, Mapping
from utils.load_brad_trace import (
    load_trace,
    create_concurrency_dataset,
)
from scheduler.base_scheduler import BaseScheduler
from simulator.simulator import QueryBank


async def submit_query_and_wait_for_result(
    connection: psycopg.AsyncConnection,
    query_rep: Union[int, str],
    sql: str,
) -> Tuple[Union[int, str], float, bool, bool]:
    error = False
    timeout = False
    async with connection.cursor() as cur:
        t = time.time()
        try:
            await cur.execute(sql)
            await cur.fetchall()
        except psycopg.errors.QueryCanceled as e:
            # this occurs in timeout
            timeout = True
        except:
            error = True
        runtime = time.time() - t
    return query_rep, runtime, timeout, error


class Executor:
    # this is the thin execution layer all users connect to instead of directly connect to the DB instance
    def __init__(
        self,
        database_kwargs: Mapping[str, Union[str, int]],
        timeout: int,
        database: str,
        scheduler: BaseScheduler,
        query_bank: Optional[QueryBank] = None,
        pause_wait_s: float = 5.0,
    ):
        try:
            loop = asyncio.get_event_loop()
        except:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
        self._loop = loop
        self.scheduler = scheduler
        self.database_kwargs = database_kwargs
        self.database = database
        self.db_conn = None
        self.timeout = timeout
        asyncio.run(self.get_connection())
        self.query_bank = query_bank
        self.pause_wait_s = pause_wait_s
        self.pending_jobs = []

    async def get_connection(self):
        self.db_conn = await psycopg.AsyncConnection.connect(**self.database_kwargs)
        acur = self.db_conn.cursor()
        timeout_ms = int(self.timeout * 1000)
        await acur.execute(f"set statement_timeout = {timeout_ms};")
        await self.db_conn.commit()
        if self.database == "Redshift":
            await acur.execute("SET enable_result_cache_for_session = OFF;")
            await self.db_conn.commit()

    def replay_one_query(
        self,
        start_time: float,
        query_rep: Optional[Union[int, str]] = None,
        query_sql: Optional[str] = None,
        query_idx: Optional[int] = None,
        baseline_run: bool = False,
    ):
        if baseline_run:
            future = asyncio.ensure_future(
                submit_query_and_wait_for_result(self.db_conn, query_rep, query_sql)
            )
            self.pending_jobs.append(future)
        else:
            (
                should_immediate_re_ingest,
                should_pause_and_re_ingest,
                scheduled_submit,
            ) = self.scheduler.ingest_query(
                start_time, query_str=query_rep, query_idx=query_idx, simulation=False
            )
            if scheduled_submit is not None:
                future = asyncio.ensure_future(
                    submit_query_and_wait_for_result(self.db_conn, query_rep, query_sql)
                )
                self.pending_jobs.append(future)
            if should_immediate_re_ingest:
                # the scheduler schedules one query at a time even if there are multiple queries in the queue,
                # so need to call again
                self.replay_one_query(start_time + 0.001)

    async def finish_all_queries(
        self,
        function_start_time: float,
        new_runtime: MutableMapping[Union[int, str], float],
        all_timeout: List[Union[int, str]],
        all_error: List[Union[int, str]],
        baseline_run: bool = False,
    ):
        current_time = time.time() - function_start_time
        if baseline_run:
            done, self.pending_jobs = await asyncio.wait(self.pending_jobs)
            for task in done:
                query_rep, runtime, timeout, error = task.result()
                new_runtime[query_rep] = runtime
                if timeout:
                    all_timeout.append(query_rep)
                if error:
                    all_error.append(query_rep)
        else:
            while (
                len(self.scheduler.queued_queries) != 0
                or len(self.scheduler.running_queries) != 0
            ):
                # make sure all queries are submitted and finished
                self.check_query_finished(
                    current_time, new_runtime, all_timeout, all_error
                )
                self.replay_one_query(current_time)
                await asyncio.sleep(self.pause_wait_s)
                current_time = time.time() - function_start_time

    def check_query_finished(
        self,
        current_time: float,
        new_runtime: MutableMapping[Union[int, str], float],
        all_timeout: List[Union[int, str]],
        all_error: List[Union[int, str]],
    ) -> bool:
        has_finished_queries = False
        if len(self.pending_jobs) != 0:
            # check if existing jobs are finished
            for task in self.pending_jobs:
                if task.done():
                    has_finished_queries = True
                    query_rep, runtime, timeout, error = task.result()
                    new_runtime[query_rep] = runtime
                    if timeout:
                        all_timeout.append(query_rep)
                    if error:
                        all_error.append(query_rep)
                        # reconnect in case of error
                        asyncio.run(self.get_connection())
                    self.pending_jobs.remove(task)
                    self.scheduler.finish_query(current_time, query_rep)
        return has_finished_queries

    async def replay_workload(
        self, directory: str, baseline_run: bool = False
    ) -> (Tuple)[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        This function replays the workload trace at the provided timestamp.
        The replayed execution will use the improved scheduling methods.
        The current implementation contains unnecessary overhead, can be improved
        :param directory: the directory containing the workload trace
        :param baseline_run: if set to True, the replay will not use client-side scheduler
        """
        function_start_time = time.time()
        all_raw_trace, all_trace = load_trace(directory, 8, concat=True)
        concurrency_df = create_concurrency_dataset(
            all_trace, engine=None, pre_exec_interval=200
        )
        concurrency_df = concurrency_df.sort_values(by=["start_time"], ascending=True)
        # the original prediction is for reference only
        original_predictions = self.scheduler.make_original_prediction(concurrency_df)
        assert len(concurrency_df) == len(original_predictions)
        new_runtime = dict()
        all_timeout = []
        all_error = []
        all_start_time = concurrency_df["start_time"].values
        all_query_idx = concurrency_df["query_idx"].values
        all_query_sql = concurrency_df["sql"].values
        for i in range(len(concurrency_df)):
            # replaying the query one-by-one
            current_time = time.time() - function_start_time
            current_query_start_time = all_start_time[i]
            while current_time < current_query_start_time - 0.5:
                # when it is not yet time to start ingest the current query according to the trace,
                has_finished_queries = self.check_query_finished(
                    current_time, new_runtime, all_timeout, all_error
                )
                if not baseline_run and has_finished_queries:
                    # reschedule the existing query when there are finished queries
                    self.replay_one_query(current_time)
                await asyncio.sleep(0.5)
                current_time = time.time() - function_start_time
            self.replay_one_query(
                current_time,
                i,
                all_query_sql[i],
                all_query_idx[i],
                baseline_run=baseline_run,
            )
        # finish all queries
        await self.finish_all_queries(
            function_start_time,
            new_runtime,
            all_timeout,
            all_error,
            baseline_run=baseline_run,
        )
        our_runtime = np.zeros(len(concurrency_df))
        for i in new_runtime:
            our_runtime[i] = new_runtime[i]
        return (
            original_predictions,
            our_runtime,
            np.asarray(all_timeout),
            np.asarray(all_error),
        )
