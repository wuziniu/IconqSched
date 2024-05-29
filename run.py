import argparse
import os.path
import asyncio
import pandas as pd
import numpy as np
import copy
import pickle as pkl
from utils.load_brad_trace import (
    create_concurrency_dataset,
    load_trace_all_version,
)
from models.single.stage import SingleStage
from models.concurrency.complex_models import ConcurrentRNN
from models.concurrency.baselines.gcn_graph_gen import generate_graph_from_trace
from models.concurrency.baselines.gcn_train import train_gcn_baseline
from scheduler.greedy_scheduler import GreedyScheduler
from scheduler.linear_programming_scheduler import LPScheduler
from simulator.simulator import Simulator
from executor.executor import Executor

np.set_printoptions(suppress=True)


def load_workload(train_test_split=True):
    all_raw_trace, all_trace = load_trace_all_version(
        args.directory, args.num_clients, concat=True
    )
    all_concurrency_df = []
    for trace in all_trace:
        concurrency_df = create_concurrency_dataset(
            trace, engine=None, pre_exec_interval=200
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


def train_concurrent_rnn():
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
    with open(f"checkpoints/{args.model_name}_stage_model.pkl", "wb") as f:
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
    )
    rnn.train(
        train_trace_df,
        eval_trace_df,
        lr=args.lr,
        loss_function=args.loss_function,
        val_on_test=args.val_on_test,
    )
    if args.target_path is not None:
        rnn.save_model(args.target_path)


def load_concurrent_rnn_stage_model():
    with open(f"checkpoints/{args.model_name}_stage_model.pkl", "rb") as f:
        ss = pkl.load(f)

    rnn = ConcurrentRNN(
        ss,
        model_prefix=args.model_name,
        input_size=len(ss.all_feature[0]) * 2 + 7,
        embedding_dim=args.embedding_dim,
        hidden_size=args.hidden_size,
        num_layers=args.num_layers,
        rnn_type=args.rnn_type,
        use_separation=args.use_separation,
    )
    rnn.load_model("checkpoints")
    return ss, rnn


def gen_trace_train_gcn_baseline():
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


def replay_workload(workload_directory, simulation, save_result_dir):
    ss, rnn = load_concurrent_rnn_stage_model()
    if args.scheduler_type == "greedy":
        scheduler = GreedyScheduler(ss, rnn)
    elif args.scheduler_type == "lp":
        scheduler = LPScheduler(ss, rnn)
    else:
        assert False, f"{args.scheduler_type} scheduler not implemented"
    if simulation:
        simulator = Simulator(scheduler)
        original_runtime, new_runtime = simulator.replay_workload(workload_directory)
        np.save(os.path.join(save_result_dir, "original_runtime_simulation"), original_runtime)
        np.save(os.path.join(save_result_dir, "new_runtime_simulation"), new_runtime)
    else:
        database_kwargs = {
            "host": args.host,
            "dbname": args.db_name,
            "port": args.port,
            "user": args.user,
            "password": args.password,
        }
        executor = Executor(database_kwargs,
                            timeout=args.timeout_s,
                            database=args.database,
                            scheduler=scheduler,
                            )
        asyncio.run(executor.replay_workload(workload_directory, args.baseline, save_result_dir))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # meta-commands
    parser.add_argument("--train_concurrent_rnn", action="store_true")
    parser.add_argument("--train_gcn_baseline", action="store_true")
    parser.add_argument("--replay_workload", action="store_true")

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
    parser.add_argument("--hidden_size", default=128, type=int)
    parser.add_argument("--num_layers", default=2, type=int)
    parser.add_argument("--lr", default=0.01, type=float)
    parser.add_argument("--loss_function", default="l1_loss", type=str)
    parser.add_argument("--val_on_test", action="store_true")

    # GCN baseline parameters
    parser.add_argument("--gcn_dataset", default="sample-plan", type=str)
    parser.add_argument("--n_run_id", default=4, type=int)
    parser.add_argument("--num_epoch", default=100, type=int)

    # Replay workload parameters
    parser.add_argument("--simulation", action="store_true")
    parser.add_argument("--baseline", action="store_true")
    parser.add_argument("--scheduler_type", default="greedy", type=str)
    parser.add_argument("--database", default="postgres", type=str)

    parser.add_argument("--timeout_s", default=200, type=int)
    parser.add_argument("--host", type=str)
    parser.add_argument("--db_name", type=str)
    parser.add_argument("--port", type=int)
    parser.add_argument("--user", type=str)
    parser.add_argument("--password", type=str)


    args = parser.parse_args()
    if args.train_concurrent_rnn:
        train_concurrent_rnn()

    if args.train_gcn_baseline:
        gen_trace_train_gcn_baseline()
