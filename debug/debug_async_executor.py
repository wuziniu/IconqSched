import time
import psycopg
import asyncio
import os
import sys

sys.path.append("../")
os.environ["PGCLIENTENCODING"] = "utf-8"
import numpy as np
from typing import Optional, Tuple, Union, MutableMapping, List, Mapping


async def submit_query_and_wait_for_result(
    connection: psycopg.AsyncConnection,
    func_start_time: float,
    query_rep: Union[int, str],
    sql: str,
) -> Tuple[Union[int, str], float, float, bool, bool]:
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
    return query_rep, t - func_start_time, runtime, timeout, error


def check_query_finished(
    pending_jobs,
) -> bool:
    has_finished_queries = False
    if len(pending_jobs) != 0:
        # check if existing jobs are finished
        for task in pending_jobs:
            if task.done():
                has_finished_queries = True
                query_rep, start_time, runtime, timeout, error = task.result()
                print(query_rep, start_time, runtime, timeout, error)
                pending_jobs.remove(task)
    return has_finished_queries


class Executor:
    # this is the thin execution layer all users connect to instead of directly connect to the DB instance
    def __init__(
        self,
        scheduler,
        database_kwargs: Mapping[str, Union[str, int]],
        timeout: int,
        database: str,
        query_bank=None,
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
    ):
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

    def finish_all_queries(
        self,
        function_start_time: float,
        new_runtime: MutableMapping[Union[int, str], float],
        all_timeout: List[Union[int, str]],
        all_error: List[Union[int, str]],
    ):
        current_time = time.time() - function_start_time
        while (
            len(self.scheduler.queued_queries) != 0
            or len(self.scheduler.running_queries) != 0
        ):
            # make sure all queries are submitted and finished
            self.check_query_finished(current_time, new_runtime, all_timeout, all_error)
            self.replay_one_query(current_time)
            time.sleep(self.pause_wait_s)
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
                    if error:
                        # reconnect in case of error
                        asyncio.run(self.get_connection())
                    new_runtime[query_rep] = runtime
                    if timeout:
                        all_timeout.append(query_rep)
                    if error:
                        all_error.append(query_rep)
                    self.pending_jobs.remove(task)
                    self.scheduler.finish_query(current_time, query_rep)
        return has_finished_queries

    async def replay_workload(self, queries):
        pending_jobs = []
        all_start_time = [1, 4, 9, 10, 14]
        function_start_time = time.time()
        for i in range(len(all_start_time)):
            current_time = time.time() - function_start_time
            current_query_start_time = all_start_time[i]
            while current_time < current_query_start_time - 0.5:
                print(i, current_time, len(pending_jobs))
                check_query_finished(pending_jobs)
                await asyncio.sleep(0.5)
                current_time = time.time() - function_start_time
            future = asyncio.ensure_future(
                submit_query_and_wait_for_result(
                    self.db_conn, function_start_time, i + 50, queries[i + 50]
                )
            )
            pending_jobs.append(future)
            print([p.done() for p in pending_jobs])
        print([p.done() for p in pending_jobs])
        done, pending = await asyncio.wait(pending_jobs)
        print([p.done() for p in done])
        print([p.result() for p in done])


def main():
    database_kwargs = {
        "host": "brad-redshift-cluster.cmdzoy6ck5ua.us-east-1.redshift.amazonaws.com",
        "dbname": "imdb_100g",
        "port": 5439,
        "user": "awsuser",
        "password": "Giftedcoconut!#4",
    }

    e = Executor(None, database_kwargs, 20, "Redshift")
    with open(
        "/Users/ziniuw/Desktop/research/Data/AWS_trace/mixed_aurora/aurora_mixed.sql",
        "r",
    ) as f:
        queries = f.readlines()

    asyncio.run(e.replay_workload(queries))


main()
