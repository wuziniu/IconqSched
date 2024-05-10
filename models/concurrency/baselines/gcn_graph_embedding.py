import os
import scipy.sparse as sp
import torch
import numpy as np
import glob


def encode_onehot(labels):
    classes = set(labels)
    classes_dict = {c: np.identity(len(classes))[i, :] for i, c in enumerate(classes)}
    labels_onehot = np.array(list(map(classes_dict.get, labels)), dtype=np.int32)
    return labels_onehot


def normalize(mx):
    """Row-normalize sparse matrix"""
    rowsum = np.array(mx.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.0
    r_mat_inv = sp.diags(r_inv)
    mx = r_mat_inv.dot(mx)
    return mx


def accuracy(output, labels):
    preds = output.max(1)[1].type_as(labels)
    correct = preds.eq(labels).double()
    correct = correct.sum()
    return correct / len(labels)


def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    """Convert a scipy sparse matrix to a torch sparse tensor."""
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(
        np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64)
    )
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse_coo_tensor(indices, values, shape, dtype=torch.float)


def load_all_data(data_dir, dataset="sample-plan", n_run_id=5):
    all_adj = []
    all_features = []
    all_labels = []
    all_root_idx_bit_map = []

    for run_id in range(n_run_id):
        files = glob.glob(data_dir + f"{dataset}-{run_id}-*-nodes.npy")
        for wid in range(len(files)):
            adj, features, labels, root_idx_bit_map, _, _ = load_data(
                dataset=f"{dataset}-{run_id}-{wid}", path=data_dir, val_len=0
            )
            all_adj.append(adj)
            all_features.append(features)
            all_labels.append(labels)
            all_root_idx_bit_map.append(root_idx_bit_map)
    return all_adj, all_features, all_labels, all_root_idx_bit_map


def load_data(dataset, path, verbose=False, val_len=0):
    if verbose:
        print("Loading {} dataset...".format(dataset))
    vmatrix = np.load(os.path.join(path, dataset + "-nodes.npy"))
    ematrix = np.load(os.path.join(path, dataset + "-edges.npy"))
    root_idx_bit_map = np.load(os.path.join(path, dataset + "-root-idx.npy"))
    adj, features, labels, idx_train, idx_val = load_data_from_matrix(
        vmatrix, ematrix, val_len
    )
    labels = labels.reshape(-1, 1)
    return adj, features, labels, root_idx_bit_map, idx_train, idx_val


def load_data_from_matrix(vmatrix, ematrix, val_len=0):
    idx_features_labels = vmatrix
    n_example = len(vmatrix)
    features = sp.csr_matrix(idx_features_labels[:, 1:-1], dtype=np.float32)
    labels = idx_features_labels[:, -1].astype(float)
    idx = np.array(idx_features_labels[:, 0], dtype=np.int32)

    idx_map = {j: i for i, j in enumerate(idx)}
    edges_unordered = ematrix[:, :-1]
    edges = np.array(
        list(map(idx_map.get, edges_unordered.flatten())), dtype=np.int32
    ).reshape(edges_unordered.shape)

    # modified begin.
    edges_value = ematrix[:, -1:]
    adj = sp.coo_matrix(
        (edges_value[:, 0], (edges[:, 0], edges[:, 1])),
        shape=(n_example, n_example),
        dtype=np.float32,
    )

    features = normalize(features)
    adj = normalize(adj + sp.eye(adj.shape[0]))

    operator_num = adj.shape[0]
    train_len = 1 - val_len
    idx_train = range(int(train_len * operator_num))
    # print("idx_train", idx_train)
    idx_val = range(int(train_len * operator_num), operator_num)
    features = torch.FloatTensor(np.array(features.todense()))
    adj = sparse_mx_to_torch_sparse_tensor(adj)
    idx_train = torch.LongTensor(idx_train)
    idx_val = torch.LongTensor(idx_val)
    labels = labels.astype(np.float32)
    labels = torch.from_numpy(labels)
    labels.unsqueeze(1)

    return adj, features, labels, idx_train, idx_val
