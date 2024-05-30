import os
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
    database_kwargs: Mapping[str, Union[str, int]],
    query_rep: Union[int, str],
    sql: str,
    query_idx: int,
    timeout_s: Optional[int] = None,
    database: Optional[str] = None
) -> Tuple[Union[int, str], int, float, bool, bool]:
    error = False
    timeout = False
    connection = await psycopg.AsyncConnection.connect(**database_kwargs)
    async with connection.cursor() as cur:
        if timeout_s:
            timeout_ms = int(timeout_s * 1000)
            await cur.execute(f"set statement_timeout = {timeout_ms};")
            await connection.commit()
        if database is not None and database == "Redshift":
            await cur.execute("SET enable_result_cache_for_session = OFF;")
            await connection.commit()
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
    return query_rep, query_idx, runtime, timeout, error


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
        debug: bool = False
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
        self.query_bank = query_bank
        self.pause_wait_s = pause_wait_s
        self.pending_jobs = []
        self.debug = debug

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
                submit_query_and_wait_for_result(self.database_kwargs, query_rep, query_sql, query_idx,
                                                 self.timeout, self.database)
            )
            self.pending_jobs.append(future)
        else:
            (
                should_immediate_re_ingest,
                should_pause_and_re_ingest,
                scheduled_submit,
            ) = self.scheduler.ingest_query(
                start_time, query_str=query_rep, query_sql=query_sql, query_idx=query_idx, simulation=False
            )
            if scheduled_submit is not None:
                query_rep, query_sql, query_idx = scheduled_submit
                future = asyncio.ensure_future(
                    submit_query_and_wait_for_result(self.database_kwargs, query_rep, query_sql, query_idx,
                                                     self.timeout, self.database)
                )
                self.pending_jobs.append(future)
            if should_immediate_re_ingest:
                # the scheduler schedules one query at a time even if there are multiple queries in the queue,
                # so need to call again
                self.replay_one_query(start_time + 0.001)

    async def finish_all_queries(
        self,
        function_start_time: float,
        e2e_runtime: MutableMapping[Union[int, str], float],
        sys_runtime: MutableMapping[Union[int, str], float],
        all_timeout: List[Union[int, str]],
        all_error: List[Union[int, str]],
        query_start_time_log: MutableMapping[Union[int, str], float],
        baseline_run: bool = False,
    ):
        current_time = time.time() - function_start_time
        if baseline_run:
            done, self.pending_jobs = await asyncio.wait(self.pending_jobs)
            for task in done:
                query_rep, query_idx, runtime, timeout, error = task.result()
                sys_runtime[query_rep] = runtime
                e2e_runtime[query_rep] = current_time - query_start_time_log[query_rep]
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
                    current_time, e2e_runtime, sys_runtime, all_timeout, all_error, query_start_time_log, baseline_run
                )
                self.replay_one_query(current_time)
                await asyncio.sleep(self.pause_wait_s)
                current_time = time.time() - function_start_time

    def check_query_finished(
        self,
        current_time: float,
        e2e_runtime: MutableMapping[Union[int, str], float],
        sys_runtime: MutableMapping[Union[int, str], float],
        all_timeout: List[Union[int, str]],
        all_error: List[Union[int, str]],
        query_start_time_log: MutableMapping[Union[int, str], float],
        is_baseline: bool = False
    ) -> bool:
        has_finished_queries = False
        if len(self.pending_jobs) != 0:
            # check if existing jobs are finished
            for task in self.pending_jobs:
                if task.done():
                    has_finished_queries = True
                    query_rep, query_idx, runtime, timeout, error = task.result()
                    sys_runtime[query_rep] = runtime
                    assert query_rep in query_start_time_log, f"no start time recorded for query {query_rep}"
                    e2e_runtime[query_rep] = current_time - query_start_time_log[query_rep]
                    if timeout:
                        all_timeout.append(query_rep)
                    if error:
                        all_error.append(query_rep)
                    self.pending_jobs.remove(task)
                    if not is_baseline:
                        self.scheduler.finish_query(current_time, query_rep)
                        if self.debug:
                            print("============================================")
                            print(self.scheduler.print_state())
                    if self.debug:
                        print(f"query {query_rep} with index {query_idx} finished with runtime {runtime}, "
                              f"timeout: {timeout}, error: {error}")
        return has_finished_queries

    def save_result(self,
                    save_result_dir: str,
                    original_predictions: List[float],
                    e2e_runtime: Mapping[int, float],
                    sys_runtime: Mapping[int, float],
                    all_timeout: List[int],
                    all_error: List[int],
                    is_baseline: bool) -> None:
        sys_exec_time = np.zeros(len(original_predictions)) - 1
        scheduler_runtime = np.zeros(len(original_predictions)) - 1
        for i in sys_runtime:
            sys_exec_time[i] = sys_runtime[i]
            scheduler_runtime[i] = e2e_runtime[i]
        original_predictions = np.asarray(original_predictions)
        all_timeout = np.asarray(all_timeout)
        all_error = np.asarray(all_error)
        np.save(os.path.join(save_result_dir, f"timeout_{self.timeout}_original_predictions"), original_predictions)
        if is_baseline:
            np.save(os.path.join(save_result_dir, f"timeout_{self.timeout}_e2e_runtime_baseline"), scheduler_runtime)
            np.save(os.path.join(save_result_dir, f"timeout_{self.timeout}_sys_exec_time_baseline"), sys_exec_time)
            np.save(os.path.join(save_result_dir, f"timeout_{self.timeout}_timeout_baseline"), all_timeout)
            np.save(os.path.join(save_result_dir, f"timeout_{self.timeout}_error_baseline"), all_error)
        else:
            np.save(os.path.join(save_result_dir, f"timeout_{self.timeout}_e2e_runtime_ours"), scheduler_runtime)
            np.save(os.path.join(save_result_dir, f"timeout_{self.timeout}_sys_exec_time_ours"), sys_exec_time)
            np.save(os.path.join(save_result_dir, f"timeout_{self.timeout}_timeout_ours"), all_timeout)
            np.save(os.path.join(save_result_dir, f"timeout_{self.timeout}_error_ours"), all_error)

    async def replay_workload(
        self, directory: str, baseline_run: bool = False,
            save_result_dir: Optional[str] = None, query_file: Optional[str] = None
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        This function replays the workload trace at the provided timestamp.
        The replayed execution will use the improved scheduling methods.
        The current implementation contains unnecessary overhead, can be improved
        :param directory: the directory containing the workload trace
        :param baseline_run: if set to True, the replay will not use client-side scheduler
        :param save_result_dir: periodically save the result to save_result_dir
        :param query_file: file that stores all queries
        """
        function_start_time = time.time()
        all_raw_trace, all_trace = load_trace(directory, 8, concat=True)
        concurrency_df = create_concurrency_dataset(
            all_trace, engine=None, pre_exec_interval=200
        )
        concurrency_df = concurrency_df.sort_values(by=["start_time"], ascending=True)
        # the original prediction is for reference only
        all_original_predictions = self.scheduler.make_original_prediction(concurrency_df)
        assert len(concurrency_df) == len(all_original_predictions)
        sys_runtime = dict()
        original_predictions = []
        e2e_runtime = dict()
        query_start_time_log = dict()
        all_timeout = []
        all_error = []
        all_start_time = concurrency_df["start_time"].values
        all_query_idx = concurrency_df["query_idx"].values
        if "sql" not in concurrency_df.columns:
            with open(query_file, "r") as f:
                queries = f.readlines()
                all_query_sql = [queries[i] for i in all_query_idx]
        else:
            all_query_sql = concurrency_df["sql"].values
        for i in range(len(concurrency_df)):
            # replaying the query one-by-one
            original_predictions.append(all_original_predictions[i])
            current_time = time.time() - function_start_time
            current_query_start_time = all_start_time[i]
            while current_time < current_query_start_time - 0.5:
                # when it is not yet time to start ingest the current query according to the trace,
                has_finished_queries = self.check_query_finished(
                    current_time, e2e_runtime, sys_runtime, all_timeout, all_error, query_start_time_log, baseline_run
                )
                if not baseline_run and has_finished_queries:
                    # reschedule the existing query when there are finished queries
                    self.replay_one_query(current_time)
                await asyncio.sleep(0.5)
                current_time = time.time() - function_start_time
            query_start_time_log[i] = current_time
            self.replay_one_query(
                current_time,
                i,
                all_query_sql[i],
                all_query_idx[i],
                baseline_run=baseline_run,
            )
            if save_result_dir is not None and (i + 1) % 100 == 0:
                self.save_result(save_result_dir,
                                 original_predictions,
                                 e2e_runtime,
                                 sys_runtime,
                                 all_timeout,
                                 all_error,
                                 baseline_run)
        # finish all queries
        await self.finish_all_queries(
            function_start_time,
            e2e_runtime,
            sys_runtime,
            all_timeout,
            all_error,
            query_start_time_log,
            baseline_run=baseline_run,
        )
        if save_result_dir is not None:
            self.save_result(save_result_dir,
                             original_predictions,
                             e2e_runtime,
                             sys_runtime,
                             all_timeout,
                             all_error,
                             baseline_run)
        sys_exec_time = np.zeros(len(concurrency_df)) - 1
        scheduler_runtime = np.zeros(len(concurrency_df)) - 1
        for i in sys_runtime:
            sys_exec_time[i] = sys_runtime[i]
            scheduler_runtime[i] = e2e_runtime[i]
        return (
            all_original_predictions,
            scheduler_runtime,
            sys_exec_time,
            np.asarray(all_timeout),
            np.asarray(all_error),
        )
