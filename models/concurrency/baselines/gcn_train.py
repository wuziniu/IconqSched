import glob
import json
import time
import bisect

import numpy as np
import torch.nn.functional as F
from tqdm import tqdm

from models.concurrency.baselines.gcn import get_model, get_optimizer
from models.concurrency.baselines.gcn_constant import args, DATAPATH
from models.concurrency.baselines.gcn_dbconnection import Database
from models.concurrency.baselines.gcn_graph_embedding import (
    load_data,
    accuracy,
    load_data_from_matrix,
    load_all_data,
)
from models.concurrency.baselines.gcn_nodeutils import (
    extract_plan,
    add_across_plan_relations,
)


def train_one_batch(model, labels, features, adj, optimizer):
    optimizer.zero_grad()
    output = model(features, adj)
    loss_train = F.mse_loss(output, labels)
    loss_train.backward()
    optimizer.step()
    return loss_train.item()


def eval_one_batch(model, labels, features, adj, root_idx_bit_map):
    output = model(features, adj)
    loss_val = F.mse_loss(output, labels).item()
    pred = output[root_idx_bit_map.astype(bool)].reshape(-1).detach().numpy()
    label = labels[root_idx_bit_map.astype(bool)].reshape(-1).numpy()
    return loss_val, pred, label


def test_one_batch(labels, features, adj, model, idx_test=None):
    model.eval()
    output = model(features, adj)
    # transfer output to ms
    # output = output * 1000
    if idx_test is not None:
        loss_test = F.mse_loss(output[idx_test], labels[idx_test])
        print("Test set results:", "loss= {:.4f}".format(loss_test.item()))
        return output[idx_test]
    else:
        loss_test = F.mse_loss(output, labels)
        print("Test set results:", "loss= {:.4f}".format(loss_test.item()))
        return output


def train_gcn_baseline(
    data_dir,
    dataset,
    n_run_id,
    num_epoch=100,
    eval_every=2,
    save_best=True,
    save_path=None,
):
    print("Loading all data graphs")
    all_adj, all_features, all_labels, all_root_idx_bit_map = load_all_data(
        data_dir, dataset, n_run_id
    )
    n_graphs = len(all_adj)
    print(f"In total, {n_graphs} number of graphs")

    model = get_model(
        feature_num=all_features[0].shape[-1],
        hidden=args.hidden,
        nclass=args.node_dim,
        dropout=args.dropout,
    )
    optimizer = get_optimizer(model=model, lr=args.lr, weight_decay=args.weight_decay)

    all_idx = np.random.permutation(n_graphs)
    train_idx = all_idx[0 : int(0.85 * n_graphs)]
    val_idx = all_idx[int(0.85 * n_graphs) :]
    best_val_loss = np.infty
    for epoch in range(num_epoch):
        model.train()
        curr_epoch_train_idx = np.random.permutation(train_idx)
        batch_loss = 0
        for i in tqdm(curr_epoch_train_idx):
            # per-batch training
            bl = train_one_batch(
                model, all_labels[i], all_features[i], all_adj[i], optimizer
            )
            batch_loss += bl
        if epoch % eval_every == 0:
            print(
                f"Epoch {epoch} ======================================================"
            )
            print(f"Training loss: {batch_loss/len(curr_epoch_train_idx)}")
            model.eval()
            val_pred = []
            val_label = []
            val_loss = 0
            for i in val_idx:
                vl, pred, label = eval_one_batch(
                    model,
                    all_labels[i],
                    all_features[i],
                    all_adj[i],
                    all_root_idx_bit_map[i],
                )
                val_loss += vl
                val_pred.append(pred)
                val_label.append(label)
            val_pred = np.maximum(np.concatenate(val_pred), 1e-3)
            val_label = np.maximum(np.concatenate(val_label), 1e-3)
            print(f"Validation loss: {val_loss / len(val_idx)}")
            abs_error = np.abs(val_pred - val_label)
            q_error = np.maximum(val_pred / val_label, val_pred / val_label)
            for p in [50, 90, 95]:
                p_a = np.percentile(abs_error, p)
                p_q = np.percentile(q_error, p)
                print(f"{p}% absolute error is {p_a}, q-error is {p_q}")
            if val_loss < best_val_loss:
                best_val_loss = val_loss


def run_train_no_upd(demo=False):
    # Step-3:
    feature_num = 3
    num_graphs = -1
    if demo:
        num_graphs = 10
    else:
        graphs = glob.glob("./pmodel_data/job/sample-plan-*")
        num_graphs = len(graphs)
    iteration_num = int(round(0.8 * num_graphs, 0))
    print("[training samples]:{}".format(iteration_num))

    for wid in range(iteration_num):
        print("[graph {}]".format(wid))

        model = get_model(
            feature_num=feature_num,
            hidden=args.hidden,
            nclass=NODE_DIM,
            dropout=args.dropout,
        )
        optimizer = get_optimizer(
            model=model, lr=args.lr, weight_decay=args.weight_decay
        )
        # Load data
        adj, features, labels, idx_train, idx_val, idx_test = load_data(
            path=DATAPATH + "/graph/", dataset="sample-plan-" + str(wid)
        )
        # print(adj.shape)
        # Model Training
        ok_times = 0
        t_total = time.time()
        for epoch in range(args.epochs):
            # print(features.shape, adj.shape)
            loss_train = train(
                epoch, labels, features, adj, idx_train, idx_val, model, optimizer
            )
            if loss_train < 0.002:
                ok_times += 1
            if ok_times >= 20:
                break

        print("Optimization Finished!")
        print("Total time elapsed: {:.4f}s".format(time.time() - t_total))
        # model validate.
        test(labels, idx_test, features=features, adj=adj, model=model)
    return iteration_num, num_graphs, model


def run_test_no_upd(iteration_num, num_graphs, model):
    for wid in range(iteration_num, num_graphs):
        # Load data
        adj, features, labels, idx_train, idx_val, idx_test = load_data(
            path=DATAPATH + "/graph/", dataset="sample-plan-" + str(wid)
        )
        # Model Testing
        t_total = time.time()
        test(labels, idx_test, features=features, adj=adj, model=model)
        print("Testing Finished!")
        print("Total time elapsed: {:.4f}s".format(time.time() - t_total))


def run_train_upd(demo=True, come_num=0):
    mp_optype = {
        "Aggregate": 0,
        "Nested Loop": 1,
        "Index Scan": 2,
        "Hash Join": 3,
        "Seq Scan": 4,
        "Hash": 5,
        "Update": 6,
    }  # operator types in the queries
    oid = 0
    min_timestamp = -1
    if demo:
        num_graphs = 4
        come_num = 1
    else:
        graphs = glob.glob("./pmodel_data/job/sample-plan-*")
        num_graphs = len(graphs)
        assert come_num == 0
    num_graphs = 4
    come_num = 1

    graphs = glob.glob("./pmodel_data/job/sample-plan-*")
    # num_graphs = len(graphs)

    # train model on a big graph composed of graph_num samples
    vmatrix = []
    ematrix = []
    feature_num = 3
    conflict_operators = {}

    for wid in range(num_graphs):
        print(wid)
        with open(DATAPATH + "/sample-plan-" + str(wid) + ".txt", "r") as f:
            for sample in f.readlines():
                sample = json.loads(sample)

                (
                    start_time,
                    node_matrix,
                    edge_matrix,
                    conflict_operators,
                    _,
                    mp_optype,
                    oid,
                    min_timestamp,
                ) = extract_plan(
                    sample, conflict_operators, mp_optype, oid, min_timestamp
                )
                # print( "OID:" + str(oid))
                vmatrix = vmatrix + node_matrix
                ematrix = ematrix + edge_matrix

        db = Database("mysql")
        knobs = db.fetch_knob()
        ematrix = add_across_plan_relations(conflict_operators, knobs, ematrix)

    model = get_model(
        feature_num=feature_num,
        hidden=args.hidden,
        nclass=NODE_DIM,
        dropout=args.dropout,
    )
    optimizer = get_optimizer(model=model, lr=args.lr, weight_decay=args.weight_decay)
    adj, features, labels, idx_train, idx_val, idx_test = load_data_from_matrix(
        np.array(vmatrix, dtype=np.float32), np.array(ematrix, dtype=np.float32)
    )

    ok_times = 0
    for epoch in range(args.epochs):
        # print(features.shape, adj.shape)
        loss_train = train(
            epoch,
            labels,
            features,
            adj,
            idx_train,
            idx_val,
            model=model,
            optimizer=optimizer,
        )
        if loss_train < 0.002:
            ok_times += 1
        if ok_times >= 20:
            break
    test(labels, idx_test, features, adj, model)
    return (
        num_graphs,
        come_num,
        model,
        adj,
        vmatrix,
        ematrix,
        mp_optype,
        oid,
        min_timestamp,
    )


def run_test_upd(
    num_graphs, come_num, model, adj, vmatrix, ematrix, mp_optype, oid, min_timestamp
):
    def predict(labels, features, adj, dh):
        model.eval()
        output = model(features, adj, dh)
        loss_test = F.mse_loss(output, labels)
        acc_test = accuracy(output, labels)
        print("Test set results:", "loss= {:.4f}".format(loss_test.item()))

    #    mp_optype = {'Aggregate': 0, 'Nested Loop': 1, 'Index Scan': 2, 'Hash Join': 3, 'Seq Scan': 4, 'Hash': 5,
    #                 'Update': 6}  # operator types in the queries
    #    oid = 0
    #    min_timestamp = -1

    #    oid = 0

    # new queries( come_num samples ) come
    # modify: new_e = []
    # change new_e -> ematrix
    conflict_operators = {}
    phi = []
    for wid in range(num_graphs, num_graphs + come_num):
        print(oid, min_timestamp)
        with open(DATAPATH + "/sample-plan-" + str(wid) + ".txt", "r") as f:
            # new query come
            for sample in f.readlines():
                # updategraph-add
                sample = json.loads(sample)

                (
                    start_time,
                    node_matrix,
                    edge_matrix,
                    conflict_operators,
                    _,
                    mp_optype,
                    oid,
                    min_timestamp,
                ) = extract_plan(
                    sample, conflict_operators, mp_optype, oid, min_timestamp
                )

                vmatrix = vmatrix + node_matrix
                ematrix = ematrix + edge_matrix

                db = Database("mysql")
                knobs = db.fetch_knob()

                ematrix = add_across_plan_relations(conflict_operators, knobs, ematrix)

                # incremental prediction
                dadj, dfeatures, dlabels, _, _, _ = load_data_from_matrix(
                    np.array(vmatrix, dtype=np.float32),
                    np.array(ematrix, dtype=np.float32),
                )

                model.eval()
                dh = model(dfeatures, dadj, None, True)

                predict(dlabels, dfeatures, adj, dh)

                for node in node_matrix:
                    bisect.insort(phi, [node[-2] + node[-1], node[0]])

                # updategraph-remove
                num = bisect.bisect(phi, [start_time, -1])
                if num > 20:  # ZXN: k = 20, num > k.
                    rmv_phi = [e[1] for e in phi[:num]]
                    phi = phi[num:]
                    vmatrix = [v for v in vmatrix if v[0] not in rmv_phi]
                    new_e = [
                        e for e in new_e if e[0] not in rmv_phi and e[1] not in rmv_phi
                    ]
                    for table in conflict_operators:
                        conflict_operators[table] = [
                            v for v in conflict_operators[table] if v[0] not in rmv_phi
                        ]
