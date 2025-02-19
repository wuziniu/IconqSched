# Adapted from BRAD.workloads.IMDB_extended.run_repeating_analytics
import argparse
import multiprocessing as mp
import time
import os
import pickle
import psycopg2
import numpy as np
import pathlib
import random
import sys
import threading
import signal
import pytz
import logging
from typing import List, Optional, Mapping
import numpy.typing as npt
from datetime import datetime


logger = logging.getLogger(__name__)
EXECUTE_START_TIME = datetime.now().astimezone(pytz.utc)
STARTUP_FAILED = "startup_failed"


def get_time_of_the_day_unsimulated(
    now: datetime, time_scale_factor: Optional[int]
) -> int:
    # Get the time of the day in minute in real-time
    assert time_scale_factor is not None, "need to specify args.time_scale_factor"
    # time_diff in minutes after scaling
    time_diff = int((now - EXECUTE_START_TIME).total_seconds() / 60 * time_scale_factor)
    time_unsimulated = time_diff % (24 * 60)  # time of the day in minutes
    return time_unsimulated


def time_in_minute_to_datetime_str(time_unsimulated: Optional[int]) -> str:
    if time_unsimulated is None:
        return "xxx"
    hour = time_unsimulated // 60
    assert hour < 24
    minute = time_unsimulated % 60
    hour_str = str(hour) if hour >= 10 else "0" + str(hour)
    minute_str = str(minute) if minute >= 10 else "0" + str(minute)
    return f"{hour_str}:{minute_str}"


def runner(
    runner_idx: int,
    start_queue: mp.Queue,
    control_semaphore: mp.Semaphore,  # type: ignore
    args,
    query_bank: List[str],
    queries: List[int],
    query_frequency: Optional[npt.NDArray] = None,
    execution_gap_dist: Optional[npt.NDArray] = None,
) -> None:
    def noop(_signal, _frame):
        pass

    signal.signal(signal.SIGINT, noop)

    # For printing out results.
    if "COND_OUT" in os.environ:
        # pylint: disable-next=import-error
        import conductor.lib as cond

        out_dir = cond.get_output_path()
    else:
        out_dir = pathlib.Path(".")

    try:
        conn = psycopg2.connect(
            host=args.host,
            port=args.port,
            database=args.schema_name,
            user=args.user,
            password=args.password,
            sslrootcert="SSLCERTIFICATE",
        )
        cur = conn.cursor()
        if args.engine == "redshift":
            cur.execute("SET enable_result_cache_for_session = OFF;")
            conn.commit()
    except:
        print(f"[RA {runner_idx}] Failed to connect to engine:")
        start_queue.put_nowait(STARTUP_FAILED)
        return

    if query_frequency is not None:
        query_frequency = query_frequency[queries]
        query_frequency = query_frequency / np.sum(query_frequency)

    exec_count = 0

    file = open(
        out_dir / f"{args.engine}_trace/repeating_olap_batch_{runner_idx}.csv",
        "w",
        encoding="UTF-8",
    )

    try:
        print(
            "timestamp,time_since_execution_s,time_of_day,query_idx,run_time_s,engine",
            file=file,
            flush=True,
        )

        prng = random.Random()
        rand_backoff = None

        logger.info(
            "[Repeating Analytics Runner %d] Queries to run: %s",
            runner_idx,
            queries,
        )
        query_order_main = queries.copy()
        prng.shuffle(query_order_main)
        query_order = query_order_main.copy()

        # Signal that we're ready to start and wait for the controller.
        print(
            f"Runner {runner_idx} is ready to start running.",
            flush=True,
            file=sys.stderr,
        )
        start_queue.put_nowait("")
        control_semaphore.acquire()  # type: ignore
        while True:
            # Note that `False` means to not block.
            should_exit = control_semaphore.acquire(False)  # type: ignore
            if should_exit:
                print(f"Runner {runner_idx} is exiting.", file=sys.stderr, flush=True)
                break

            if execution_gap_dist is not None:
                now = datetime.now().astimezone(pytz.utc)
                time_unsimulated = get_time_of_the_day_unsimulated(
                    now, args.time_scale_factor
                )
                wait_for_s = execution_gap_dist[
                    int(time_unsimulated / (60 * 24) * len(execution_gap_dist))
                ]
                time.sleep(wait_for_s)
            elif args.avg_gap_s is not None:
                # Wait times are normally distributed if execution_gap_dist is not provided.
                wait_for_s = prng.gauss(args.avg_gap_s, args.avg_gap_std_s)
                if wait_for_s < 0.0:
                    wait_for_s = 0.0
                time.sleep(wait_for_s)

            if query_frequency is not None:
                qidx = prng.choices(queries, list(query_frequency))[0]
            else:
                if len(query_order) == 0:
                    query_order_main = queries.copy()
                    prng.shuffle(query_order_main)
                    query_order = query_order_main.copy()

                qidx = query_order.pop()
            logger.debug("Executing qidx: %d", qidx)
            query = query_bank[qidx]

            try:
                now = datetime.now().astimezone(pytz.utc)
                if args.time_scale_factor is not None:
                    time_unsimulated = get_time_of_the_day_unsimulated(
                        now, args.time_scale_factor
                    )
                    time_unsimulated_str = time_in_minute_to_datetime_str(
                        time_unsimulated
                    )
                else:
                    time_unsimulated_str = "xxx"
                if args.engine == "aurora" or args.engine == "postgres":
                    cur.execute("SET statement_timeout to 1000000;")
                    conn.commit()
                elif args.engine == "redshift":
                    cur.execute("set statement_timeout = 1000000;")
                    conn.commit()
                start = time.time()
                try:
                    cur.execute(query)
                    cur.fetchall()
                except psycopg2.errors.QueryCanceled as e:
                    print(f"query {qidx} timeout")
                except:
                    continue
                end = time.time()
                print(
                    "{},{},{},{},{},{}".format(
                        now,
                        (now - EXECUTE_START_TIME).total_seconds(),
                        time_unsimulated_str,
                        qidx,
                        end - start,
                        args.engine,
                    ),
                    file=file,
                    flush=True,
                )

                if exec_count % 20 == 0:
                    # To avoid data loss if this script crashes.
                    os.fsync(file.fileno())

                exec_count += 1
                if rand_backoff is not None:
                    print(
                        f"[RA {runner_idx}] Continued after transient errors.",
                        flush=True,
                        file=sys.stderr,
                    )
                    rand_backoff = None

            except Exception as ex:
                print(
                    "Unexpected query error:",
                    ex,
                    flush=True,
                    file=sys.stderr,
                )
                conn = psycopg2.connect(
                    host=args.host,
                    port=args.port,
                    database=args.schema_name,
                    user=args.user,
                    password=args.password,
                    sslrootcert="SSLCERTIFICATE",
                )
                cur = conn.cursor()
                if args.engine == "redshift":
                    cur.execute("SET enable_result_cache_for_session = OFF;")
                    conn.commit()

    finally:
        os.fsync(file.fileno())
        file.close()
        conn.close()
        print(f"Runner {runner_idx} has exited.", flush=True, file=sys.stderr)


def run_warmup(args, query_bank: List[str], queries: List[int]):
    conn = psycopg2.connect(
        host=args.host,
        port=args.port,
        database=args.schema_name,
        user=args.user,
        password=args.password,
        sslrootcert="SSLCERTIFICATE",
    )
    cur = conn.cursor()
    if args.engine == "redshift":
        cur.execute("SET enable_result_cache_for_session = OFF;")
        conn.commit()

    # For printing out results.
    if "COND_OUT" in os.environ:
        # pylint: disable-next=import-error
        import conductor.lib as cond

        out_dir = cond.get_output_path()
    else:
        out_dir = pathlib.Path(".")

    try:
        print(
            f"Starting warmup pass (will run {args.run_warmup_times} times)...",
            file=sys.stderr,
            flush=True,
        )
        with open(
            out_dir / f"{args.engine}_trace/repeating_olap_batch_warmup.csv",
            "w",
            encoding="UTF-8",
        ) as file:
            print("timestamp,query_idx,run_time_s,engine", file=file)
            for _ in range(args.run_warmup_times):
                for idx, qidx in enumerate(queries):
                    try:
                        query = query_bank[qidx]
                        now = datetime.now().astimezone(pytz.utc)
                        if args.engine == "aurora" or args.engine == "postgres":
                            cur.execute("SET statement_timeout to 1000000;")
                            conn.commit()
                        elif args.engine == "redshift":
                            cur.execute("set statement_timeout = 1000000;")
                            conn.commit()
                        start = time.time()
                        cur.execute(query)
                        cur.fetchall()
                        end = time.time()
                        run_time_s = end - start
                        print(
                            "Warmed up {} of {}. Run time (s): {}".format(
                                idx + 1, len(queries), run_time_s
                            ),
                            file=sys.stderr,
                            flush=True,
                        )
                        print(
                            "{},{},{},{}".format(
                                now,
                                qidx,
                                run_time_s,
                                args.engine if args.engine is not None else "unknown",
                            ),
                            file=file,
                            flush=True,
                        )
                    except Exception as ex:
                        print(
                            "Unexpected query error:",
                            ex,
                            flush=True,
                            file=sys.stderr,
                        )
                        conn = psycopg2.connect(
                            host=args.host,
                            port=args.port,
                            database=args.schema_name,
                            user=args.user,
                            password=args.password,
                            sslrootcert="SSLCERTIFICATE",
                        )
                        cur = conn.cursor()
                        if args.engine == "redshift":
                            cur.execute("SET enable_result_cache_for_session = OFF;")
                            conn.commit()
    finally:
        conn.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--brad-host", type=str, default="localhost")
    parser.add_argument("--brad-port", type=int, default=6583)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-front-ends", type=int, default=1)
    parser.add_argument("--run-warmup", action="store_true")
    parser.add_argument("--host", type=str)
    parser.add_argument("--port", type=int, default=5432)
    parser.add_argument("--user", type=str)
    parser.add_argument("--password", type=str)
    parser.add_argument(
        "--run-simulation",
        action="store_true",
        help="Run the simulation instead of actual execution.",
    )
    parser.add_argument(
        "--wait-for-execute-sim",
        action="store_true",
        help="Waiting for execution in simulation?",
    )
    parser.add_argument(
        "--query-runtime-path",
        type=str,
        default=None,
        help="path to the query runtime numpy file",
    )
    parser.add_argument(
        "--run-warmup-times",
        type=int,
        default=1,
        help="Run the warmup query list this many times.",
    )
    parser.add_argument(
        "--cstr-var",
        type=str,
        help="Set to connect via ODBC instead of the BRAD client (for use with other baselines).",
    )
    parser.add_argument(
        "--query-bank-file", type=str, required=True, help="Path to a query bank."
    )
    parser.add_argument(
        "--query-frequency-path",
        type=str,
        default=None,
        help="path to the frequency to draw each query in query bank",
    )
    parser.add_argument(
        "--num-query-path",
        type=str,
        default=None,
        help="Path to the distribution of number of queries for each period of a day",
    )
    parser.add_argument(
        "--num-client-path",
        type=str,
        default=None,
        help="Path to the distribution of number of clients for each period of a day",
    )
    parser.add_argument("--num-clients", type=int, default=1)
    parser.add_argument("--client-offset", type=int, default=0)
    parser.add_argument("--avg-gap-s", type=float)
    parser.add_argument("--avg-gap-std-s", type=float, default=0.5)
    parser.add_argument(
        "--gap-dist-path",
        type=str,
        default=None,
        help="Path to the distribution regarding the number of concurrent queries",
    )
    parser.add_argument(
        "--time-scale-factor",
        type=int,
        default=100,
        help="trace 1s of simulation as X seconds in real-time to match the num-concurrent-query",
    )
    parser.add_argument("--query-indexes", type=str)
    parser.add_argument(
        "--brad-direct",
        action="store_true",
        help="Set to connect directly to Aurora via BRAD's config.",
    )
    parser.add_argument(
        "--config-file",
        type=str,
        help="The BRAD config file (if --brad-direct is used).",
    )
    parser.add_argument(
        "--schema-name",
        type=str,
        help="The schema name to use, if connecting directly.",
    )
    parser.add_argument(
        "--engine", type=str, help="The engine to use, if connecting directly."
    )
    parser.add_argument("--run-for-s", type=int, help="If set, run for this long.")
    args = parser.parse_args()

    with open(args.query_bank_file, "r", encoding="UTF-8") as file:
        query_bank = [line.strip() for line in file]

    if args.query_frequency_path is not None and os.path.exists(
        args.query_frequency_path
    ):
        query_frequency = np.load(args.query_frequency_path)
        assert len(query_frequency) == len(
            query_bank
        ), "query_frequency size does not match total number of queries"
    else:
        query_frequency = None

    if (
        args.gap_dist_path is not None
        and os.path.exists(args.gap_dist_path)
        and args.time_scale_factor is not None
    ):
        # we can only set the num_concurrent_query trace in presence of time_scale_factor
        execution_gap_dist = np.load(args.gap_dist_path)
    else:
        execution_gap_dist = None

    if (
        args.num_client_path is not None
        and os.path.exists(args.num_client_path)
        and args.time_scale_factor is not None
    ):
        # we can only set the num_concurrent_query trace in presence of time_scale_factor
        with open(args.num_client_path, "rb") as f:
            num_client_trace = pickle.load(f)
    else:
        num_client_trace = None

    if args.query_indexes is None:
        queries = list(range(len(query_bank)))
    else:
        queries = list(map(int, args.query_indexes.split(",")))

    for qidx in queries:
        assert qidx < len(query_bank)
        assert qidx >= 0

    if args.run_warmup:
        run_warmup(args, query_bank, queries)
        return

    # Our control protocol is as follows.
    # - Runner processes write to their `start_queue` when they have finished
    #   setting up and are ready to start running. They then wait on the control
    #   semaphore.
    # - The control process blocks and waits on each `start_queue` to ensure
    #   runners can start together (if needed).
    # - The control process signals the control semaphore twice. Once to tell a
    #   runner to start, once to tell it to stop.
    # - If there is an error, a runner is free to exit as long as they have
    #   written to `start_queue`.
    mgr = mp.Manager()
    start_queue = [mgr.Queue() for _ in range(args.num_clients)]
    # N.B. `value = 0` since we use this for synchronization, not mutual exclusion.
    # pylint: disable-next=no-member
    control_semaphore = [mgr.Semaphore(value=0) for _ in range(args.num_clients)]

    processes = []
    for idx in range(args.num_clients):
        p = mp.Process(
            target=runner,
            args=(
                idx,
                start_queue[idx],
                control_semaphore[idx],
                args,
                query_bank,
                queries,
                query_frequency,
                execution_gap_dist,
            ),
        )
        p.start()
        processes.append(p)

    print("Waiting for startup...", flush=True)
    one_startup_failed = False
    for i in range(args.num_clients):
        msg = start_queue[i].get()
        if msg == STARTUP_FAILED:
            one_startup_failed = True

    if one_startup_failed:
        print("At least one runner failed to start up. Aborting the experiment.")
        for i in range(args.num_clients):
            # Ideally we should be able to release twice atomically.
            control_semaphore[i].release()
            control_semaphore[i].release()
        for p in processes:
            p.join()
        print("Abort complete.")
        return

    global EXECUTE_START_TIME  # pylint: disable=global-statement
    EXECUTE_START_TIME = datetime.now().astimezone(
        pytz.utc
    )  # pylint: disable=global-statement

    if num_client_trace is not None:
        assert args.time_scale_factor is not None, "Need to set --time-scale-factor"
        assert args.run_for_s is not None, "Need to set --run-for-s"
        print("Telling client no. 0 to start.", flush=True)
        control_semaphore[0].release()
        num_running_client = 1

        finished_one_day = True
        curr_day_start_time = datetime.now().astimezone(pytz.utc)
        for time_of_day in num_client_trace:
            if time_of_day == 0:
                continue
            # at this time_of_day start/shut-down more clients
            time_in_s = time_of_day / args.time_scale_factor
            now = datetime.now().astimezone(pytz.utc)
            curr_time_in_s = (now - curr_day_start_time).total_seconds()
            total_exec_time_in_s = (now - EXECUTE_START_TIME).total_seconds()
            if args.run_for_s <= total_exec_time_in_s:
                finished_one_day = False
                break
            if args.run_for_s - total_exec_time_in_s <= (time_in_s - curr_time_in_s):
                wait_time = args.run_for_s - total_exec_time_in_s
                if wait_time > 0:
                    time.sleep(wait_time)
                finished_one_day = False
                break
            time.sleep(time_in_s - curr_time_in_s)
            num_client_required = min(num_client_trace[time_of_day], args.num_clients)
            if num_client_required > num_running_client:
                # starting additional clients
                for add_client in range(num_running_client, num_client_required):
                    print(
                        "Telling client no. {} to start.".format(add_client), flush=True
                    )
                    control_semaphore[add_client].release()
                    num_running_client += 1
            elif num_running_client > num_client_required:
                # shutting down clients
                for delete_client in range(num_running_client, num_client_required, -1):
                    print(
                        "Telling client no. {} to stop.".format(delete_client - 1),
                        flush=True,
                    )
                    control_semaphore[delete_client - 1].release()
                    num_running_client -= 1
        now = datetime.now().astimezone(pytz.utc)
        total_exec_time_in_s = (now - EXECUTE_START_TIME).total_seconds()
        if finished_one_day:
            print(
                f"Finished executing one day of workload in {total_exec_time_in_s}s, will ignore the rest of "
                f"pre-set execution time {args.run_for_s}s"
            )
        else:
            print(
                f"Executed ended but unable to finish executing the trace of a full day within {args.run_for_s}s"
            )

    else:
        print("Telling all {} clients to start.".format(args.num_clients), flush=True)
        for i in range(args.num_clients):
            control_semaphore[i].release()

    if args.run_for_s is not None and num_client_trace is None:
        print(
            "Waiting for {} seconds...".format(args.run_for_s),
            flush=True,
            file=sys.stderr,
        )
        time.sleep(args.run_for_s)
    elif num_client_trace is None:
        # Wait until requested to stop.
        print(
            "Repeating analytics waiting until requested to stop... (hit Ctrl-C)",
            flush=True,
            file=sys.stderr,
        )
        should_shutdown = threading.Event()

        def signal_handler(_signal, _frame):
            should_shutdown.set()

        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGTERM, signal_handler)

        should_shutdown.wait()

    print("Stopping all clients...", flush=True, file=sys.stderr)
    for i in range(args.num_clients):
        # Note that in most cases, one release will have already run. This is OK
        # because downstream runners will not hang if there is a unconsumed
        # semaphore value.
        control_semaphore[i].release()
        control_semaphore[i].release()

    print("Waiting for the clients to complete...", flush=True, file=sys.stderr)
    for p in processes:
        p.join()

    for idx, p in enumerate(processes):
        print(f"Runner {idx} exit code:", p.exitcode, flush=True, file=sys.stderr)

    print("Done repeating analytics!", flush=True, file=sys.stderr)


if __name__ == "__main__":
    # On Unix platforms, the default way to start a process is by forking, which
    # is not ideal (we do not want to duplicate this process' file
    # descriptors!).
    mp.set_start_method("spawn")
    main()
