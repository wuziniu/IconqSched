import torch
import numpy as np
import torch.nn.functional as F
from tqdm import tqdm

from models.concurrency.complex_models import q_loss_func
from models.concurrency.baselines.gcn_v2 import get_model, get_optimizer
from models.concurrency.baselines.gcn_constant import args
from models.concurrency.baselines.gcn_graph_embedding import load_all_data


def train_one_batch(model, labels, features, edge_idx, edge_weight, optimizer, loss_func="qloss"):
    optimizer.zero_grad()
    output = model(features, edge_idx, edge_weight)
    if loss_func == "qloss":
        loss_train = q_loss_func(output, labels)
    else:
        loss_train = F.mse_loss(output, labels)
    loss_train.backward()
    optimizer.step()
    return loss_train.item()


def eval_one_batch(model, labels, features, edge_idx, edge_weight, root_idx_bit_map):
    output = model(features, edge_idx, edge_weight)
    loss_val = F.mse_loss(output, labels).item()
    pred = output[root_idx_bit_map.astype(bool)].reshape(-1).detach().numpy()
    label = labels[root_idx_bit_map.astype(bool)].reshape(-1).numpy()
    return loss_val, pred, label


def test_one_batch(labels, features, edge_idx, edge_weight, model, idx_test=None):
    model.eval()
    output = model(features, edge_idx, edge_weight,)
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
    all_features, all_edge_idx, all_edge_weight, all_labels, all_root_idx_bit_map = load_all_data(
        data_dir, dataset, n_run_id
    )
    n_graphs = len(all_features)
    print(f"In total, {n_graphs} number of graphs")

    model = get_model(
        feature_num=all_features[0].shape[-1],
        hidden=args.hidden,
        nclass=args.node_dim,
        nlayers=args.n_layers,
        dropout=args.dropout,
    )
    optimizer = get_optimizer(model=model, lr=args.lr, weight_decay=args.weight_decay)

    all_idx = np.random.permutation(n_graphs)
    train_idx = all_idx[0: int(0.85 * n_graphs)]
    val_idx = all_idx[int(0.85 * n_graphs):]
    best_val_loss = np.infty
    for epoch in range(num_epoch):
        model.train()
        curr_epoch_train_idx = np.random.permutation(train_idx)
        batch_loss = 0
        for i in tqdm(curr_epoch_train_idx):
            # per-batch training
            bl = train_one_batch(
                model, all_labels[i], all_features[i], all_edge_idx[i], all_edge_weight[i], optimizer, loss_func="mse"
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
                    all_edge_idx[i],
                    all_edge_weight[i],
                    all_root_idx_bit_map[i],
                )
                val_loss += vl
                val_pred.append(pred)
                val_label.append(label)
            val_pred = np.maximum(np.concatenate(val_pred), 1e-3)
            val_label = np.maximum(np.concatenate(val_label), 1e-3)
            print(f"Validation loss: {val_loss / len(val_idx)}")
            abs_error = np.abs(val_pred - val_label)
            q_error = np.maximum(val_pred / val_label, val_label / val_pred)
            for p in [50, 90, 95]:
                p_a = np.percentile(abs_error, p)
                p_q = np.percentile(q_error, p)
                print(f"{p}% absolute error is {p_a}, q-error is {p_q}")
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                if save_best and save_path is not None:
                    torch.save(model.state_dict(), save_path)
    if not save_best and save_path is not None:
        torch.save(model.state_dict(), save_path)
