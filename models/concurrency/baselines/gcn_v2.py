# a better implementation of gcn.py
import numpy as np
import torch
from typing import List, Union, Optional, Tuple
from torch import optim
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv


class GCN(nn.Module):
    def __init__(self, nfeat, nhid, nclass, nlayers, dropout):
        super(GCN, self).__init__()
        self.nlayers = nlayers
        self.gc1 = SAGEConv(nfeat, nhid)
        self.gc2 = SAGEConv(nhid, nhid)
        self.fc = nn.Sequential(nn.Linear(nhid, nclass), nn.Linear(nclass, 1))
        self.dropout = dropout

    def forward(self, x, edge_idx, edge_weight, dh=None, embed=False):
        x = self.gc1(x, edge_idx)
        x = F.dropout(x, self.dropout, training=self.training)
        x = self.gc2(x, edge_idx)
        x = F.dropout(x, self.dropout, training=self.training)
        if embed:
            return x
        if dh is not None:
            x = x + dh
        x = self.fc(x)
        return x

    def online_inference(
        self,
        existing_query_features: List[np.ndarray],
        existing_query_concur_features: List[Optional[torch.Tensor]],
        existing_pre_info_length: List[int],
        queued_query_features: List[np.ndarray],
        existing_start_time: List[float],
        current_time: float,
        next_finish_idx: Optional[Union[int, List[int]]] = None,
        next_finish_time: Optional[Union[float, List[float]]] = None,
        get_next_finish: bool = False,
        get_next_finish_running_performance: bool = False,
    ) -> Tuple[torch.Tensor, List[torch.Tensor], torch.Tensor]:
        global_x, global_pre_info_length = featurize_queries_complex_online(
            existing_query_features,
            existing_query_concur_features,
            existing_pre_info_length,
            queued_query_features,
            existing_start_time,
            current_time,
            next_finish_idx,
            next_finish_time,
            get_next_finish,
            get_next_finish_running_performance,
            use_pre_exec_info=self.use_pre_exec_info,
        )
        predictions = self.model(global_x, None, global_pre_info_length, False)
        return predictions, global_x, global_pre_info_length


def get_model(feature_num, hidden, nclass, nlayers, dropout):
    model = GCN(
        nfeat=feature_num, nhid=hidden, nclass=nclass, nlayers=nlayers, dropout=dropout
    )
    return model


def get_optimizer(model, lr, weight_decay):
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    return optimizer
