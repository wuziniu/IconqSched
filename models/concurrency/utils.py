import torch
from torch.utils.data import Dataset
import torch.nn as nn


class QueryFeatureDataset(Dataset):
    def __init__(self, feature, label):
        self.feature = feature
        self.label = label

    def __len__(self):
        return len(self.label)

    def __getitem__(self, idx):
        return_feature = []
        for f in self.feature:
            return_feature.append(f[idx])
        return tuple(return_feature), self.label[idx]


class SimpleNet(nn.Module):
    def __init__(self, input_dim, hidden_dim=64, layers=3):
        super(SimpleNet, self).__init__()
        model = []
        model.append(nn.Linear(input_dim, hidden_dim))
        model.append(nn.ReLU())
        for i in range(layers):
            model.append(nn.Linear(hidden_dim, hidden_dim))
            model.append(nn.ReLU())
        model.append(nn.Linear(hidden_dim, 1))
        self.model = nn.Sequential(*model)

    def forward(self, x):
        pred = self.model(x)
        return torch.maximum(pred, torch.tensor(0.01))
