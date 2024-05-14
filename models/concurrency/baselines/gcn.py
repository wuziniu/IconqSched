import math
import torch
from torch import optim
import torch.nn as nn
import torch.nn.functional as F

# this code is adapted from https://github.com/zhouxh19/workload-performance


class GraphConvolution(nn.Module):
    """
    Simple GCN layer, similar to https://arxiv.org/abs/1609.02907
    """

    def __init__(self, in_features, out_features, bias=True):
        super(GraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.FloatTensor(in_features, out_features))
        if bias:
            self.bias = nn.Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter("bias", None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1.0 / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, x, adj):
        support = torch.mm(x, self.weight)
        output = torch.spmm(adj, support)
        if self.bias is not None:
            return output + self.bias
        else:
            return output

    def __repr__(self):
        return (
            self.__class__.__name__
            + " ("
            + str(self.in_features)
            + " -> "
            + str(self.out_features)
            + ")"
        )


class GCN(nn.Module):
    def __init__(self, nfeat, nhid, nclass, nlayers, dropout):
        super(GCN, self).__init__()
        self.nlayers = nlayers
        self.gc1 = GraphConvolution(nfeat, nhid)
        self.gc2 = GraphConvolution(nhid, nhid)
        self.gc3 = GraphConvolution(nhid, nclass)
        self.fc = nn.Linear(nclass, 1)
        self.dropout = dropout

    def forward(self, x, adj, dh=None, embed=False):
        x = self.gc1(x, adj)
        x = F.dropout(x, self.dropout, training=self.training)
        x = self.gc2(x, adj)
        x = F.dropout(x, self.dropout, training=self.training)
        x = self.gc3(x, adj)
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
