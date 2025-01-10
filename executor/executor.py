import os
import pandas as pd
import psycopg
import logging
import asyncio
import time
import numpy as np
from typing import Optional, Tuple, Union, MutableMapping, List, Mapping
from utils.load_trace import (
    load_trace,
    create_concurrency_dataset,
)
from scheduler.base_scheduler import BaseScheduler
from scheduler.pgm_scheduler import PGMScheduler
from scheduler.qshuffler_scheduler import QShuffler
from simulator.simulator import QueryBank


async def submit_query_and_wait_for_result(
    database_kwargs: Mapping[str, Union[str, int]],
    query_rep: Union[int, str],
    sql: str,
    query_idx: int,
    timeout_s: Optional[float] = None,
    database: Optional[str] = None,
    max_retry: int = 3,
) -> Tuple[Union[int, str], int, float, bool, bool]:
    error = False
    timeout = False
    for _ in range(max_retry):
        try:
            connection = await psycopg.AsyncConnection.connect(**database_kwargs)
            async with connection.cursor() as cur:
                if timeout_s:
                    if timeout_s <= 0:
                        return query_rep, query_idx, 0.0, True, error
                    timeout_ms = int(timeout_s * 1000)
                    await cur.execute(f"set statement_timeout = {timeout_ms};")
                    await connection.commit()
                if database is not None and database == "redshift":
                    await cur.execute("SET enable_result_cache_for_session = OFF;")
                    await connection.commit()
                t = time.time()
                try:
                    await cur.execute(sql)
                    await cur.fetchall()
                except psycopg.errors.QueryCanceled as e:
                    # this occurs in timeout
                    timeout = True
                except Exception as e:
                    print(
                        f"Executing query {query_rep} with index {query_idx}, encountered error: ",
                        e,
                    )
                    error = True
                runtime = time.time() - t
            return query_rep, query_idx, runtime, timeout, error
        except Exception as e:
            print("Error: ", e)
            print("Trying to reconnect and re-execute")
    assert (
        False
    ), f"Connection failed after {max_retry}. This is a bug with psycopg.AsyncConnection."


class Executor:
    # this is the thin execution layer all users connect to instead of directly connect to the DB instance
    def __init__(
        self,
        database_kwargs: Mapping[str, Union[str, int]],
        timeout: int,
        database: str,
        scheduler: Optional[Union[BaseScheduler, PGMScheduler, QShuffler]],
        query_bank: Optional[QueryBank] = None,
        pause_wait_s: float = 5.0,
        debug: bool = False,
        logger: Optional[logging.Logger] = None,
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
        if database == "redshift":
            os.environ["PGCLIENTENCODING"] = "utf-8"
        self.db_conn = None
        self.timeout = timeout
        self.query_bank = query_bank
        self.pause_wait_s = pause_wait_s
        self.pending_jobs = []
        self.debug = debug
        self.logger = logger
        self.num_clients = 0
        self.query_exec_start_time = dict()

    async def get_connection_async(self):
        self.db_conn = await psycopg.AsyncConnection.connect(**self.database_kwargs)
        acur = self.db_conn.cursor()
        timeout_ms = int(self.timeout * 1000)
        await acur.execute(f"set statement_timeout = {timeout_ms};")
        await self.db_conn.commit()
        if self.database == "redshift":
            await acur.execute("SET enable_result_cache_for_session = OFF;")
            await self.db_conn.commit()

    def get_connection_sync(self) -> psycopg.cursor:
        db_conn = psycopg.connect(**self.database_kwargs)
        cur = db_conn.cursor()
        timeout_ms = int(self.timeout * 1000)
        cur.execute(f"set statement_timeout = {timeout_ms};")
        db_conn.commit()
        if self.database == "redshift":
            cur.execute("SET enable_result_cache_for_session = OFF;")
            db_conn.commit()
        return cur

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
                submit_query_and_wait_for_result(
                    self.database_kwargs,
                    query_rep,
                    query_sql,
                    query_idx,
                    self.timeout,
                    self.database,
                )
            )
            self.pending_jobs.append(future)
            self.query_exec_start_time[query_rep] = start_time
        else:
            (
                should_immediate_re_ingest,
                should_pause_and_re_ingest,
                scheduled_submit,
            ) = self.scheduler.ingest_query(
                start_time,
                query_str=query_rep,
                query_sql=query_sql,
                query_idx=query_idx,
                simulation=False,
            )
            if scheduled_submit is not None:
                query_rep, query_sql, query_idx, queueing_time = scheduled_submit
                future = asyncio.ensure_future(
                    submit_query_and_wait_for_result(
                        self.database_kwargs,
                        query_rep,
                        query_sql,
                        query_idx,
                        self.timeout - queueing_time,
                        self.database,
                    )
                )
                self.pending_jobs.append(future)
                self.query_exec_start_time[query_rep] = start_time
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
                    current_time,
                    e2e_runtime,
                    sys_runtime,
                    all_timeout,
                    all_error,
                    query_start_time_log,
                    baseline_run,
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
        is_baseline: bool = False,
    ) -> bool:
        has_finished_queries = False
        if len(self.pending_jobs) != 0:
            # check if existing jobs are finished
            for task in self.pending_jobs:
                if task.done():
                    has_finished_queries = True
                    query_rep, query_idx, runtime, timeout, error = task.result()
                    sys_runtime[query_rep] = runtime
                    assert (
                        query_rep in query_start_time_log
                    ), f"no start time recorded for query {query_rep}"
                    e2e_runtime[query_rep] = (
                        current_time - query_start_time_log[query_rep]
                    )
                    queueing_time = e2e_runtime[query_rep] - sys_runtime[query_rep]
                    if timeout:
                        all_timeout.append(query_rep)
                    if error:
                        all_error.append(query_rep)
                    self.pending_jobs.remove(task)
                    if not is_baseline:
                        self.scheduler.finish_query(current_time, query_rep)
                        if self.debug:
                            if self.logger is None:
                                print(
                                    "===================[Scheduler State]========================="
                                )
                                self.scheduler.print_state()
                            else:
                                self.logger.info(
                                    "======================[Scheduler State]======================"
                                )
                                self.scheduler.print_state()
                    if self.debug:
                        if self.logger is None:
                            print(
                                f"[[[[[[[[[[[query {query_rep} with index {query_idx} finished with "
                                f"runtime: {runtime}, queueing_time: {queueing_time},"
                                f"timeout: {timeout}, error: {error}]]]]]]]]]]]]"
                            )
                        else:
                            self.logger.info(
                                f"[[[[[[[[[[query {query_rep} with index {query_idx} finished with "
                                f"runtime: {runtime}, queueing_time: {queueing_time},"
                                f"timeout: {timeout}, error: {error}]]]]]]]]]]"
                            )
        return has_finished_queries

    def save_result(
        self,
        save_result_dir: str,
        original_predictions: List[float],
        e2e_runtime: Mapping[int, float],
        sys_runtime: Mapping[int, float],
        all_timeout: List[int],
        all_error: List[int],
        is_baseline: bool,
    ) -> None:
        if not os.path.exists(save_result_dir):
            os.mkdir(save_result_dir)
        sys_exec_time = np.zeros(len(original_predictions)) - 1
        scheduler_runtime = np.zeros(len(original_predictions)) - 1
        for i in sys_runtime:
            sys_exec_time[i] = sys_runtime[i]
            scheduler_runtime[i] = e2e_runtime[i]
        original_predictions = np.asarray(original_predictions)
        all_timeout = np.asarray(all_timeout)
        all_error = np.asarray(all_error)
        np.save(
            os.path.join(
                save_result_dir,
                f"{self.database}_timeout_{self.timeout}_original_predictions",
            ),
            original_predictions,
        )
        if is_baseline:
            np.save(
                os.path.join(
                    save_result_dir,
                    f"{self.database}_timeout_{self.timeout}_e2e_runtime_baseline",
                ),
                scheduler_runtime,
            )
            np.save(
                os.path.join(
                    save_result_dir,
                    f"{self.database}_timeout_{self.timeout}_sys_exec_time_baseline",
                ),
                sys_exec_time,
            )
            np.save(
                os.path.join(
                    save_result_dir,
                    f"{self.database}_timeout_{self.timeout}_timeout_baseline",
                ),
                all_timeout,
            )
            np.save(
                os.path.join(
                    save_result_dir,
                    f"{self.database}_timeout_{self.timeout}_error_baseline",
                ),
                all_error,
            )
        else:
            np.save(
                os.path.join(
                    save_result_dir,
                    f"{self.database}_timeout_{self.timeout}_e2e_runtime_ours",
                ),
                scheduler_runtime,
            )
            np.save(
                os.path.join(
                    save_result_dir,
                    f"{self.database}_timeout_{self.timeout}_sys_exec_time_ours",
                ),
                sys_exec_time,
            )
            np.save(
                os.path.join(
                    save_result_dir,
                    f"{self.database}_timeout_{self.timeout}_timeout_ours",
                ),
                all_timeout,
            )
            np.save(
                os.path.join(
                    save_result_dir,
                    f"{self.database}_timeout_{self.timeout}_error_ours",
                ),
                all_error,
            )

    def save_result_as_df(
        self,
        save_result_dir: str,
        all_query_idx: List[int],
        all_query_no: List[int],
        e2e_runtime: Mapping[int, float],
        sys_runtime: Mapping[int, float],
        query_start_time_log: Mapping[int, float],
        all_timeout: List[int],
        all_error: List[int],
        is_baseline: bool,
        warmup_run: bool = False,
        return_df: bool = True,
        save_prefix: Optional[str] = "",
    ) -> Optional[pd.DataFrame]:
        assert len(all_query_no) == len(all_query_idx)
        query_start_time = np.zeros(len(all_query_no)) - 1
        sys_exec_time = np.zeros(len(all_query_no)) - 1
        scheduler_runtime = np.zeros(len(all_query_no)) - 1
        timeout_per_query = []
        error_per_query = []
        for i, q_no in enumerate(all_query_no):
            if q_no in sys_runtime:
                sys_exec_time[i] = sys_runtime[q_no]
            if q_no in e2e_runtime:
                scheduler_runtime[i] = e2e_runtime[q_no]
            if q_no in query_start_time_log:
                query_start_time[i] = query_start_time_log[q_no]
            if i in all_timeout:
                timeout_per_query.append(True)
            else:
                timeout_per_query.append(False)
            if i in all_error:
                error_per_query.append(True)
            else:
                error_per_query.append(False)
        df = pd.DataFrame(
            {
                "index": all_query_no,
                "query_idx": all_query_idx,
                "run_time_s": list(scheduler_runtime),
                "exec_time": list(sys_exec_time),
                "time_since_execution_s": list(query_start_time),
                "g_offset_since_start_s": list(query_start_time),
                "timeout": timeout_per_query,
                "error": error_per_query,
            }
        )
        if warmup_run:
            df.to_csv(
                os.path.join(
                    save_result_dir,
                    f"{save_prefix}_timeout_{self.timeout}_warmup_run.csv",
                ),
                index=False,
            )
        elif is_baseline:
            df.to_csv(
                os.path.join(
                    save_result_dir,
                    f"{save_prefix}timeout_{self.timeout}_baseline.csv",
                ),
                index=False,
            )
        else:
            df.to_csv(
                os.path.join(
                    save_result_dir,
                    f"{save_prefix}timeout_{self.timeout}_ours.csv",
                ),
                index=False,
            )
        if return_df:
            return df

    def warmup_run(
        self,
        query_file: str,
        save_result_dir: str,
        selected_query_idx_path: Optional[str] = None,
    ) -> pd.DataFrame:
        function_start_time = time.time()
        cur = self.get_connection_sync()
        with open(query_file, "r") as f:
            queries = f.readlines()
        if selected_query_idx_path is not None:
            all_possible_query_idx = np.load(selected_query_idx_path)
            queries = queries[all_possible_query_idx]
        else:
            all_possible_query_idx = np.arange(len(queries))
        sys_runtime = dict()
        all_query_idx = []
        all_query_no = []
        e2e_runtime = dict()
        query_start_time_log = dict()
        all_timeout = []
        all_error = []
        for i, sql in enumerate(queries):
            timeout = False
            error = False
            t = time.time()
            start_time = t - function_start_time
            try:
                cur.execute(sql)
                cur.fetchall()
            except psycopg.errors.QueryCanceled as e:
                # this occurs in timeout
                timeout = True
                cur = self.get_connection_sync()
            except:
                error = True
                cur = self.get_connection_sync()
            runtime = time.time() - t
            sys_runtime[i] = runtime
            e2e_runtime[i] = runtime
            all_query_no.append(i)
            all_query_idx.append(all_possible_query_idx[i])
            all_timeout.append(timeout)
            all_error.append(error)
            query_start_time_log[i] = start_time
            if i % 40 == 0:
                self.save_result_as_df(
                    save_result_dir,
                    all_query_idx,
                    all_query_no,
                    e2e_runtime,
                    sys_runtime,
                    query_start_time_log,
                    all_timeout,
                    all_error,
                    warmup_run=True,
                    is_baseline=True,
                    return_df=False,
                )
        df = self.save_result_as_df(
            save_result_dir,
            all_query_idx,
            all_query_no,
            e2e_runtime,
            sys_runtime,
            query_start_time_log,
            all_timeout,
            all_error,
            warmup_run=True,
            is_baseline=True,
            return_df=True,
        )
        return df

    async def run_k_client_in_parallel(
        self,
        query_file: str,
        num_clients: int = 5,
        baseline_run: bool = True,
        save_result_dir: Optional[str] = None,
        gap_s: float = 1.0,
        exec_for_s: Optional[float] = 3600.0,
        selected_query_idx_path: Optional[str] = None,
        seed: int = 0,
    ) -> pd.DataFrame:
        """
        This function conduct a close-loop run of k clients.
        Specifically, we create k clients, each issuing queries continuously to the system.
        Whenever the client receives the result of previous query, it issues the next one
        :param query_file: file that stores all queries
        :param num_clients: number of clients to execute in parallel
        :param baseline_run: if set to True, the replay will not use client-side scheduler
        :param save_result_dir: periodically save the result to save_result_dir
        :param gap_s: the gap in second before next query
        :param exec_for_s: for long do we execute for
        :param selected_query_idx_path: provide a numpy array of selected queries and only issue queries from them
        :param seed: random seed
        """
        np.random.seed(seed)
        self.num_clients = num_clients
        with open(query_file, "r") as f:
            queries = f.readlines()
        if selected_query_idx_path is not None:
            all_possible_query_idx = np.load(selected_query_idx_path)
        else:
            all_possible_query_idx = np.arange(len(queries))
        function_start_time = time.time()
        sys_runtime = dict()
        all_query_idx = []
        all_query_no = []
        e2e_runtime = dict()
        query_start_time_log = dict()
        all_timeout = []
        all_error = []
        if exec_for_s is None:
            exec_for_s = np.infty
        current_time = time.time() - function_start_time
        curr_query_no = 0
        recently_save = False
        while current_time < exec_for_s:
            await asyncio.sleep(gap_s)
            current_time = time.time() - function_start_time
            has_finished_queries = self.check_query_finished(
                current_time,
                e2e_runtime,
                sys_runtime,
                all_timeout,
                all_error,
                query_start_time_log,
                baseline_run,
            )
            if not baseline_run and has_finished_queries:
                # reschedule the existing query when there are finished queries
                self.replay_one_query(current_time)
            if len(self.pending_jobs) < num_clients:
                selected_query_idx = int(
                    all_possible_query_idx[
                        np.random.randint(len(all_possible_query_idx))
                    ]
                )
                selected_query_sql = queries[selected_query_idx]
                all_query_idx.append(selected_query_idx)
                all_query_no.append(curr_query_no)
                current_time = time.time() - function_start_time
                query_start_time_log[curr_query_no] = current_time
                self.replay_one_query(
                    current_time,
                    curr_query_no,
                    selected_query_sql,
                    selected_query_idx,
                    baseline_run=baseline_run,
                )
                curr_query_no += 1
            if (
                save_result_dir is not None
                and not recently_save
                and (curr_query_no + 1) % 50 == 0
            ):
                self.save_result_as_df(
                    save_result_dir,
                    all_query_idx,
                    all_query_no,
                    e2e_runtime,
                    sys_runtime,
                    query_start_time_log,
                    all_timeout,
                    all_error,
                    is_baseline=baseline_run,
                    return_df=False,
                    save_prefix=f"clients_{self.num_clients}_",
                )
                recently_save = True
            if (curr_query_no + 1) % 50 == 1:
                recently_save = False
        df = self.save_result_as_df(
            save_result_dir,
            all_query_idx,
            all_query_no,
            e2e_runtime,
            sys_runtime,
            query_start_time_log,
            all_timeout,
            all_error,
            is_baseline=baseline_run,
            return_df=True,
            save_prefix=f"clients_{self.num_clients}_",
        )
        return df

    async def replay_workload(
        self,
        directory: str,
        baseline_run: bool = False,
        save_result_dir: Optional[str] = None,
        query_file: Optional[str] = None,
        start_idx: Optional[int] = 0,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        This function replays the workload trace at the provided timestamp.
        The replayed execution will use the improved scheduling methods.
        The current implementation contains unnecessary overhead, can be improved
        :param directory: the directory containing the workload trace
        :param baseline_run: if set to True, the replay will not use client-side scheduler
        :param save_result_dir: periodically save the result to save_result_dir
        :param query_file: file that stores all queries
        :param start_idx: where to start in this trace
        """
        if directory.endswith(".csv"):
            # just one single csv file
            all_trace = pd.read_csv(directory)
            save_prefix = directory.split("/")[-1].split(".csv")[0] + "_"
        else:
            all_raw_trace, all_trace = load_trace(directory, 8, concat=True)
            save_prefix = ""
        all_trace = all_trace[all_trace["run_time_s"] > 0]
        concurrency_df = create_concurrency_dataset(
            all_trace, engine=None, pre_exec_interval=200.0
        )
        concurrency_df = concurrency_df.sort_values(by=["start_time"], ascending=True)
        # the original prediction is for reference only
        print(f"Starting the replay of workload at index {start_idx}")
        concurrency_df = concurrency_df[start_idx:]

        if isinstance(self.scheduler, BaseScheduler):
            all_original_predictions = self.scheduler.make_original_prediction(
                concurrency_df
            )
            assert len(concurrency_df) == len(all_original_predictions)
        else:
            all_original_predictions = np.zeros(len(concurrency_df))

        sys_runtime = dict()
        original_predictions = []
        e2e_runtime = dict()
        query_start_time_log = dict()
        all_query_no = []
        all_timeout = []
        all_error = []
        all_start_time = concurrency_df["start_time"].values
        all_start_time = all_start_time - all_start_time[0]
        all_query_idx = concurrency_df["query_idx"].values
        if "sql" not in concurrency_df.columns:
            with open(query_file, "r") as f:
                queries_text = f.read()
            queries = queries_text.split(";")[:-1]
            all_query_sql = [queries[int(i)].strip() + ';' for i in all_query_idx]
        else:
            all_query_sql = concurrency_df["sql"].values
        curr_query_no = start_idx
        function_start_time = time.time()
        for i in range(len(concurrency_df)):
            # replaying the query one-by-one
            original_predictions.append(all_original_predictions[i])
            current_time = time.time() - function_start_time
            current_query_start_time = all_start_time[i]
            while current_time < current_query_start_time - 0.5:
                # when it is not yet time to start ingest the current query according to the trace,
                has_finished_queries = self.check_query_finished(
                    current_time,
                    e2e_runtime,
                    sys_runtime,
                    all_timeout,
                    all_error,
                    query_start_time_log,
                    baseline_run,
                )
                if not baseline_run and has_finished_queries:
                    # reschedule the existing query when there are finished queries
                    self.replay_one_query(current_time)
                await asyncio.sleep(0.5)
                current_time = time.time() - function_start_time
            query_start_time_log[curr_query_no] = current_time
            self.replay_one_query(
                current_time,
                curr_query_no,
                all_query_sql[i],
                all_query_idx[i],
                baseline_run=baseline_run,
            )
            all_query_no.append(curr_query_no)
            curr_query_no += 1
            if save_result_dir is not None and (i + 1) % 50 == 0:
                self.save_result_as_df(
                    save_result_dir,
                    all_query_idx[: i + 1],
                    all_query_no,
                    e2e_runtime,
                    sys_runtime,
                    self.query_exec_start_time,
                    all_timeout,
                    all_error,
                    is_baseline=baseline_run,
                    return_df=False,
                    save_prefix=save_prefix,
                )
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
            self.save_result_as_df(
                save_result_dir,
                all_query_idx,
                all_query_no,
                e2e_runtime,
                sys_runtime,
                self.query_exec_start_time,
                all_timeout,
                all_error,
                is_baseline=baseline_run,
                return_df=False,
                save_prefix=save_prefix,
            )
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
