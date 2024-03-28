import torch
from torch.utils.data import Dataset
import torch.nn as nn


class QueryFeatureDataset(Dataset):
    def __init__(self, feature, label):
        self.feature = torch.vstack(feature).T
        self.feature = self.feature.type('torch.FloatTensor')
        print(self.feature.shape)
        self.label = label.type('torch.FloatTensor')


    def __len__(self):
        return len(self.label)

    def __getitem__(self, idx):
        return self.feature[idx], self.label[idx]


class QueryFeatureSeparatedDataset(Dataset):
    def __init__(self, feature, label, x1_idx, x2_idx, x3_idx):
        self.feature = torch.vstack(feature).T
        self.feature = self.feature.type('torch.FloatTensor')
        self.label = label
        self.x1_idx = x1_idx
        self.x2_idx = x2_idx
        self.x3_idx = x3_idx

    def __len__(self):
        return len(self.label)

    def __getitem__(self, idx):
        return self.feature[idx][:, self.x1_idx], self.feature[idx][:, self.x2_idx], self.feature[idx][:, self.x3_idx], self.label[idx]


class SimpleNet(nn.Module):
    def __init__(self, input_dim, hidden_dim=64, layers=3):
        super(SimpleNet, self).__init__()
        model = []
        model.append(nn.Linear(input_dim, hidden_dim))
        model.append(nn.ReLU())
        for i in range(layers):
            model.append(nn.Linear(hidden_dim, hidden_dim))
            model.append(nn.ReLU())
        model.append(nn.Dropout(0.9))
        model.append(nn.Linear(hidden_dim, 1))
        self.model = nn.Sequential(*model)

    def forward(self, x):
        pred = self.model(x)
        return torch.maximum(pred, torch.tensor(0.01))
