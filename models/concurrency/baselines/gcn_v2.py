# a better implementation of gcn.py
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
        self.fc = nn.Sequential(
            nn.Linear(nhid, nclass),
            nn.Linear(nclass, 1))
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


def get_model(feature_num, hidden, nclass, nlayers, dropout):
    model = GCN(nfeat=feature_num, nhid=hidden, nclass=nclass, nlayers=nlayers, dropout=dropout)
    return model


def get_optimizer(model, lr, weight_decay):
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    return optimizer
