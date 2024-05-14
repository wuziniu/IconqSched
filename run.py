import argparse
import pandas as pd
import numpy as np
import copy
from utils.load_brad_trace import (
    create_concurrency_dataset,
    load_trace_all_version,
)
from models.single.stage import SingleStage
from models.concurrency.complex_models import ConcurrentRNN
from models.concurrency.baselines.gcn_graph_gen import generate_graph_from_trace
from models.concurrency.baselines.gcn_train import train_gcn_baseline
np.set_printoptions(suppress=True)


def train_concurrent_rnn():
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

    ss = SingleStage(
        use_size=args.true_card,
        use_log=args.use_log,
        true_card=args.true_card,
        use_table_features=args.use_table_features,
        use_table_selectivity=args.use_table_selectivity,
    )
    df = ss.featurize_data(train_trace_df, args.parsed_queries_path)
    ss.train(df)

    rnn = ConcurrentRNN(
        ss,
        input_size=len(ss.all_feature[0]) * 2 + 7,
        embedding_dim=args.embedding_dim,
        hidden_size=args.hidden_size,
        num_layers=args.num_layers,
        rnn_type=args.rnn_type,
        use_seperation=args.use_separation,
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


def train_gcn_baseline():
    generate_graph_from_trace(args.directory, args.parsed_queries_path, args.gcn_graph_path)
    train_gcn_baseline(args.gcn_graph_path,
                       args.gcn_dataset,
                       args.n_run_id,
                       num_epoch=args.num_epoch,
                       eval_every=5,
                       save_best=True,
                       save_path=args.target_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # meta-commands
    parser.add_argument("--train_concurrent_rnn", action="store_true")
    parser.add_argument("--train_gcn_baseline", action="store_true")
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


    args = parser.parse_args()
    if args.train_concurrent_rnn:
        train_concurrent_rnn()
