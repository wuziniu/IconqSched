import os
import scipy.sparse as sp
import torch
import numpy as np
import glob


def load_all_data(data_dir, dataset="sample-plan", n_run_id=5):
    all_edge_idx = []
    all_edge_weight = []
    all_features = []
    all_labels = []
    all_root_idx_bit_map = []

    for run_id in range(n_run_id):
        files = glob.glob(data_dir + f"{dataset}-{run_id}-*-nodes.npy")
        for wid in range(len(files)):
            features, edge_idx, edge_weight, labels, root_idx_bit_map = load_data(
                dataset=f"{dataset}-{run_id}-{wid}", path=data_dir
            )
            all_edge_idx.append(edge_idx)
            all_edge_weight.append(edge_weight)
            all_features.append(features)
            all_labels.append(labels)
            all_root_idx_bit_map.append(root_idx_bit_map)
    return all_features, all_edge_idx, all_edge_weight, all_labels, all_root_idx_bit_map


def load_data(dataset, path, verbose=False):
    if verbose:
        print("Loading {} dataset...".format(dataset))
    vmatrix = np.load(os.path.join(path, dataset + "-nodes.npy"))
    ematrix = np.load(os.path.join(path, dataset + "-edges.npy"))
    root_idx_bit_map = np.load(os.path.join(path, dataset + "-root-idx.npy"))
    features, edge_idx, edge_weight, labels = load_data_from_matrix(vmatrix, ematrix)
    labels = labels.reshape(-1, 1)
    return features, edge_idx, edge_weight, labels, root_idx_bit_map


def load_data_from_matrix(vmatrix, ematrix):
    features = torch.from_numpy(vmatrix[:, 1:-1]).type(torch.FloatTensor)
    labels = torch.from_numpy(vmatrix[:, -1]).type(torch.FloatTensor)
    labels.unsqueeze(1)

    edge_idx = torch.from_numpy(ematrix[:, :-1].T).type(torch.int64)
    edge_weight = torch.from_numpy(ematrix[:, -1]).type(torch.FloatTensor)
    return features, edge_idx, edge_weight, labels
