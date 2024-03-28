import torch
import torch.nn as nn


class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dim=64, layers=3):
        super(MLP, self).__init__()
        model = []
        model.append(nn.Linear(input_dim, hidden_dim))
        model.append(nn.ReLU())
        for i in range(layers):
            model.append(nn.Linear(hidden_dim, hidden_dim))
            model.append(nn.ReLU())
        model.append(nn.Dropout(0.9))
        model.append(nn.Linear(hidden_dim, 1))
        self.model = nn.Sequential(*model)
        self.is_train = True

    def forward(self, x1, x2, x3):
        y1 = self.model(x1, x2)
        if self.is_train:
            pred = self.model(y1, x3)
        else:
            pred = y1
        return torch.maximum(pred, torch.tensor(0.01))
