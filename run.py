import argparse
import os.path
import asyncio
from typing import Union, Tuple, Optional
import pandas as pd
import numpy as np
import copy
import pickle as pkl
from utils.load_trace import (
    create_concurrency_dataset,
    load_trace_all_version,
    load_all_csv_from_dir,
)
from parser.parse_plan import get_query_plans
from models.single.stage import SingleStage
from models.concurrency.complex_models import ConcurrentRNN
from models.concurrency.baselines.qshuffler_estimator import QEstimator
from models.concurrency.baselines.gcn_graph_gen import generate_graph_from_trace
from models.concurrency.baselines.gcn_train import train_gcn_baseline
from scheduler.greedy_scheduler import GreedyScheduler
from scheduler.pgm_scheduler import PGMScheduler
from scheduler.qshuffler_scheduler import QShuffler
from scheduler.linear_programming_scheduler import LPScheduler
from simulator.simulator import Simulator
from executor.executor import Executor
from utils.logging import create_custom_logger
from workloads.workload_tools.mimic_trace import pre_process_snowset, TraceManager

np.set_printoptions(suppress=True)


def load_workload(
    train_test_split: bool = True,
    load_csv: bool = True,
) -> Union[pd.DataFrame, Tuple[pd.DataFrame, pd.DataFrame]]:
    if load_csv:
        all_trace = load_all_csv_from_dir(args.directory)
    else:
        _, all_trace = load_trace_all_version(
            args.directory, args.num_clients, concat=True
        )
    all_concurrency_df = []
    for trace in all_trace:
        concurrency_df = create_concurrency_dataset(
            trace, engine=None, pre_exec_interval=None
        )
        all_concurrency_df.append(concurrency_df)
    concurrency_df = pd.concat(all_concurrency_df, ignore_index=True)

    if train_test_split:
        np.random.seed(0)
        train_idx = np.random.choice(
            len(concurrency_df), size=int(0.8 * len(concurrency_df)), replace=False
        )
        test_idx = [i for i in range(len(concurrency_df)) if i not in train_idx]
        train_trace_df = copy.deepcopy(concurrency_df.iloc[train_idx])
        eval_trace_df = concurrency_df.iloc[test_idx]
        eval_trace_df = copy.deepcopy(
            eval_trace_df[eval_trace_df["num_concurrent_queries"] > 0]
        )
        print(len(train_trace_df), len(eval_trace_df))
        return train_trace_df, eval_trace_df
    else:
        return concurrency_df


def minic_snowset_workload(
    trace_path: str, save_result_dir: str, query_bank_path: str
) -> None:
    trace = pd.read_csv(trace_path)
    if (
        "g_offset_since_start_s" not in trace.columns
        or "run_time_s" not in trace.columns
    ):
        trace = pre_process_snowset(trace)
        trace.to_csv(trace_path, index=False)

    ss, rnn = load_concurrent_rnn_stage_model()
    tm = TraceManager(
        ss,
        rnn,
        query_bank_path,
        database=args.database,
        debug=args.debug,
        stride=args.stride,
    )
    minic_trace = tm.simulate_trace_iteratively(trace_path, max_iter=args.max_iter)
    minic_trace.to_csv(save_result_dir, index=False)


def train_concurrent_rnn() -> None:
    train_trace_df, eval_trace_df = load_workload(train_test_split=True)
    ss = SingleStage(
        use_size=args.true_card,
        use_log=args.use_log,
        true_card=args.true_card,
        use_table_features=args.use_table_features,
        use_table_selectivity=args.use_table_selectivity,
    )
    df = ss.featurize_data(train_trace_df, args.parsed_queries_path)
    ss.train(df)
    with open(
        os.path.join(args.target_path, f"{args.model_name}_stage_model.pkl"), "wb"
    ) as f:
        pkl.dump(ss, f)

    rnn = ConcurrentRNN(
        ss,
        model_prefix=args.model_name,
        input_size=len(ss.all_feature[0]) * 2 + 7,
        embedding_dim=args.embedding_dim,
        hidden_size=args.hidden_size,
        num_layers=args.num_layers,
        rnn_type=args.rnn_type,
        use_separation=args.use_separation,
        ignore_short_running=args.ignore_short_running,
        short_running_threshold=args.short_running_threshold,
    )
    rnn.train(
        train_trace_df,
        eval_trace_df,
        lr=args.lr,
        loss_function=args.loss_function,
        val_on_test=args.val_on_test,
        epochs=args.epochs,
    )
    if args.target_path is not None:
        rnn.save_model(args.target_path)


def load_concurrent_rnn_stage_model(
    scheduler_type: str = "greedy",
) -> Tuple[SingleStage, Optional[Union[ConcurrentRNN, QEstimator]]]:
    with open(
        os.path.join(args.target_path, f"{args.model_name}_stage_model.pkl"), "rb"
    ) as f:
        ss = pkl.load(f)

    model = None
    if scheduler_type == "greedy" or scheduler_type == "lp":
        model = ConcurrentRNN(
            ss,
            model_prefix=args.model_name,
            input_size=len(ss.all_feature[0]) * 2 + 7,
            embedding_dim=args.embedding_dim,
            hidden_size=args.hidden_size,
            num_layers=args.num_layers,
            rnn_type=args.rnn_type,
            use_separation=args.use_separation,
            ignore_short_running=args.ignore_short_running,
            short_running_threshold=args.short_running_threshold,
        )
        model.load_model(args.target_path)
    elif scheduler_type == "qshuffler":
        with open(
            os.path.join(args.target_path, f"{args.model_name}_qestimator.pkl"), "rb"
        ) as f:
            model = pkl.load(f)
    return ss, model


def gen_trace_train_gcn_baseline() -> None:
    generate_graph_from_trace(
        args.directory, args.parsed_queries_path, args.gcn_graph_path
    )
    train_gcn_baseline(
        args.gcn_graph_path,
        args.gcn_dataset,
        args.n_run_id,
        num_epoch=args.num_epoch,
        eval_every=5,
        save_best=True,
        save_path=args.target_path,
    )


def warmup_run(query_bank_path: str) -> None:
    database_kwargs = {
        "host": args.host,
        "dbname": args.db_name,
        "port": args.port,
        "user": args.user,
        "password": args.password,
    }
    executor = Executor(
        database_kwargs, timeout=args.timeout_s, database=args.database, scheduler=None
    )
    executor.warmup_run(
        query_bank_path, args.save_result_dir, args.selected_query_idx_path
    )


def replay_workload(
    workload_directory: str, save_result_dir: str, query_bank_path: str, baseline: bool
) -> None:
    if not os.path.exists(save_result_dir):
        os.makedirs(save_result_dir)
    if args.debug:
        verbose_log_dir = os.path.join(save_result_dir, "verbose_logs")
        if not os.path.exists(verbose_log_dir):
            os.mkdir(verbose_log_dir)
        if baseline:
            log_name = "baseline"
        else:
            log_name = "ours"
        run_id = np.random.randint(100000)
        log_file_path = os.path.join(verbose_log_dir, f"{log_name}_{run_id}.log")
        verbose_logger = create_custom_logger(log_name, log_file_path)
        print(f"Debug log save to: {log_file_path}")
    else:
        verbose_logger = None
    if args.scheduler_type == "greedy":
        ss, model = load_concurrent_rnn_stage_model(args.scheduler_type)
        scheduler = GreedyScheduler(
            ss,
            model,
            debug=args.debug,
            logger=verbose_logger,
            ignore_short_running=args.ignore_short_running,
            starve_penalty=args.starve_penalty,
            alpha=args.alpha,
            short_running_threshold=args.short_running_threshold,
            steps_into_future=args.steps_into_future,
        )
    elif args.scheduler_type == "lp":
        ss, model = load_concurrent_rnn_stage_model(args.scheduler_type)
        scheduler = LPScheduler(ss, model)
    elif args.scheduler_type == "qshuffler":
        ss, model = load_concurrent_rnn_stage_model(args.scheduler_type)
        scheduler = QShuffler(
            ss,
            model,
            debug=args.debug,
            logger=verbose_logger,
            ignore_short_running=args.ignore_short_running,
            short_running_threshold=args.short_running_threshold,
            lookahead=args.lookahead,
            mpl=args.qshuffler_mpl,
        )
    elif args.scheduler_type == "pgm":
        ss, model = load_concurrent_rnn_stage_model(args.scheduler_type)
        scheduler = PGMScheduler(
            ss,
            debug=args.debug,
            logger=verbose_logger,
            ignore_short_running=args.ignore_short_running,
            short_running_threshold=args.short_running_threshold,
            use_memory=args.use_memory,
            admission_threshold=args.admission_threshold,
            consider_top_k=args.consider_top_k,
        )
    else:
        scheduler = None
        assert args.baseline, f"{args.scheduler_type} scheduler not implemented and is not baseline run"
    if args.simulation:
        simulator = Simulator(scheduler)
        original_runtime, new_runtime = simulator.replay_workload(workload_directory)
        np.save(
            os.path.join(save_result_dir, "original_runtime_simulation"),
            original_runtime,
        )
        np.save(os.path.join(save_result_dir, "new_runtime_simulation"), new_runtime)
    else:
        database_kwargs = {
            "host": args.host,
            "dbname": args.db_name,
            "port": args.port,
            "user": args.user,
            "password": args.password,
        }
        executor = Executor(
            database_kwargs,
            timeout=args.timeout_s,
            database=args.database,
            scheduler=scheduler,
            debug=args.debug,
            logger=verbose_logger,
        )
        asyncio.run(
            executor.replay_workload(
                workload_directory,
                baseline,
                save_result_dir,
                query_bank_path,
                start_idx=args.start_idx,
            )
        )


def run_k_client_in_parallel(
    query_bank_path: str,
    num_clients: int,
    save_result_dir: str,
    selected_query_idx_path: Optional[str] = None,
) -> None:
    if args.debug:
        verbose_log_dir = os.path.join(save_result_dir, "verbose_logs")
        if not os.path.exists(verbose_log_dir):
            os.mkdir(verbose_log_dir)
        if args.baseline:
            log_name = "baseline"
        else:
            log_name = "ours"
        run_id = np.random.randint(100000)
        log_file_path = os.path.join(verbose_log_dir, f"{log_name}_{run_id}.log")
        verbose_logger = create_custom_logger(log_name, log_file_path)
        print(f"Debug log save to: {log_file_path}")
    else:
        verbose_logger = None
    if args.scheduler_type == "greedy":
        ss, rnn = load_concurrent_rnn_stage_model()
        scheduler = GreedyScheduler(
            ss,
            rnn,
            debug=args.debug,
            logger=verbose_logger,
            ignore_short_running=args.ignore_short_running,
            starve_penalty=args.starve_penalty,
            alpha=args.alpha,
            short_running_threshold=args.short_running_threshold,
            steps_into_future=args.steps_into_future,
        )
    elif args.scheduler_type == "lp":
        ss, rnn = load_concurrent_rnn_stage_model()
        scheduler = LPScheduler(ss, rnn)
    else:
        scheduler = None
        assert args.baseline, f"{args.scheduler_type} scheduler not implemented and not in a baseline run"

    database_kwargs = {
        "host": args.host,
        "dbname": args.db_name,
        "port": args.port,
        "user": args.user,
        "password": args.password,
    }
    executor = Executor(
        database_kwargs,
        timeout=args.timeout_s,
        database=args.database,
        scheduler=scheduler,
        debug=args.debug,
        logger=verbose_logger,
    )
    asyncio.run(
        executor.run_k_client_in_parallel(
            query_bank_path,
            num_clients,
            args.baseline,
            save_result_dir,
            selected_query_idx_path=selected_query_idx_path,
            exec_for_s=args.exec_for_s,
            seed=args.seed,
        )
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # meta-commands
    parser.add_argument("--parse_explain", action="store_true")
    parser.add_argument("--train_concurrent_rnn", action="store_true")
    parser.add_argument("--train_gcn_baseline", action="store_true")
    parser.add_argument("--replay_workload", action="store_true")
    parser.add_argument("--replay_workload_ours_and_baseline", action="store_true")
    parser.add_argument("--warmup_run", action="store_true")
    parser.add_argument("--run_k_client_in_parallel", action="store_true")
    parser.add_argument("--minic_snowset_workload", action="store_true")

    # path information
    parser.add_argument("--gcn_graph_path", type=str)
    parser.add_argument("--target_path", type=str)

    # query featurization parameters
    parser.add_argument("--use_size", action="store_true")
    parser.add_argument("--use_log", action="store_true")
    parser.add_argument("--true_card", action="store_true")
    parser.add_argument("--rnn_type", default="lstm", type=str)
    parser.add_argument("--use_separation", action="store_true")
    parser.add_argument("--use_table_features", action="store_true")
    parser.add_argument("--use_table_selectivity", action="store_true")

    # load dataset
    parser.add_argument("--num_clients", default=8, type=int)
    parser.add_argument("--parsed_queries_path", type=str)
    parser.add_argument("--directory", type=str)

    # LSTM hyperparameters
    parser.add_argument("--model_name", default="postgres", type=str)
    parser.add_argument("--embedding_dim", default=128, type=int)
    parser.add_argument("--hidden_size", default=256, type=int)
    parser.add_argument("--num_layers", default=2, type=int)
    parser.add_argument("--lr", default=0.001, type=float)
    parser.add_argument("--epochs", default=100, type=int)
    parser.add_argument("--loss_function", default="q_loss", type=str)
    parser.add_argument("--val_on_test", action="store_true")

    # GCN baseline parameters
    parser.add_argument("--gcn_dataset", default="sample-plan", type=str)
    parser.add_argument("--n_run_id", default=4, type=int)
    parser.add_argument("--num_epoch", default=100, type=int)

    # PGM scheduler parameter
    parser.add_argument("--use_memory", action="store_true")
    parser.add_argument("--admission_threshold", type=int, default=1000)
    parser.add_argument("--consider_top_k", type=int, default=2)

    # QShuffler parameter
    parser.add_argument("--lookahead", type=int, default=10)
    parser.add_argument("--qshuffler_mpl", type=int, default=5)

    # Replay workload parameters
    parser.add_argument("--scheduler_type", default="greedy", type=str)
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--ignore_short_running", action="store_true")
    parser.add_argument("--steps_into_future", type=int, default=2)
    parser.add_argument("--alpha", type=float, default=0.2)
    parser.add_argument("--start_idx", type=int, default=0)
    parser.add_argument("--short_running_threshold", type=float, default=5.0)
    parser.add_argument("--starve_penalty", type=float, default=0.5)
    parser.add_argument("--simulation", action="store_true")
    parser.add_argument("--seed", type=int, default=24)
    parser.add_argument("--baseline", action="store_true")
    parser.add_argument("--exec_for_s", type=int, default=24 * 3600)
    parser.add_argument("--num_clients_list", type=str, default=None)
    parser.add_argument("--selected_query_idx_path", type=str)
    parser.add_argument("--database", default="postgres", type=str)
    parser.add_argument("--save_result_dir", type=str)
    parser.add_argument("--query_bank_path", type=str)
    parser.add_argument("--timeout_s", default=200, type=int)
    parser.add_argument("--host", type=str)
    parser.add_argument("--db_name", type=str)
    parser.add_argument("--port", type=int)
    parser.add_argument("--user", type=str)
    parser.add_argument("--password", type=str)
    parser.add_argument("--stride", type=int, default=4)
    parser.add_argument("--max_iter", type=int, default=100)

    args = parser.parse_args()
    if args.parse_explain:
        database_kwargs = {
            "host": args.host,
            "database": args.db_name,
            "port": args.port,
            "user": args.user,
            "password": args.password,
        }
        get_query_plans(args.query_bank_path, None, args.database, args.save_result_dir, database_kwargs)

    if args.train_concurrent_rnn:
        train_concurrent_rnn()

    if args.train_gcn_baseline:
        gen_trace_train_gcn_baseline()

    if args.minic_snowset_workload:
        minic_snowset_workload(
            args.directory, args.save_result_dir, args.query_bank_path
        )

    if args.warmup_run:
        warmup_run(args.query_bank_path)

    if args.run_k_client_in_parallel:
        if args.num_clients_list is not None:
            num_clients_list = list(map(int, args.num_clients_list.split(",")))
            for num_clients in num_clients_list:
                print(num_clients)
                run_k_client_in_parallel(
                    args.query_bank_path,
                    num_clients,
                    args.save_result_dir,
                    args.selected_query_idx_path,
                )
        else:
            run_k_client_in_parallel(
                args.query_bank_path,
                args.num_clients,
                args.save_result_dir,
                args.selected_query_idx_path,
            )

    if args.replay_workload:
        replay_workload(
            args.directory, args.save_result_dir, args.query_bank_path, args.baseline
        )

    elif args.replay_workload_ours_and_baseline:
        replay_workload(
            args.directory, args.save_result_dir, args.query_bank_path, True
        )
        replay_workload(
            args.directory, args.save_result_dir, args.query_bank_path, False
        )
