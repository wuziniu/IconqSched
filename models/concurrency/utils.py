import copy
import torch
from torch.utils.data import Dataset
import torch.nn as nn


class QueryFeatureDataset(Dataset):
    def __init__(self, feature, label):
        self.feature = torch.vstack(feature).T
        self.feature = self.feature.type("torch.FloatTensor")
        print(self.feature.shape)
        self.label = label.type("torch.FloatTensor")

    def __len__(self):
        return len(self.label)

    def __getitem__(self, idx):
        return self.feature[idx], self.label[idx]


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


def pre_info_train_test_seperation(concurrency_df):
    # picking out all the query without post_info as testing data and the rest as training data
    train_idx = []
    test_idx = []
    concur_info_train = concurrency_df["concur_info_train"].values
    concur_info_full = concurrency_df["concur_info"].values
    for i in range(len(concurrency_df)):
        if len(concur_info_train[i]) == len(concur_info_full[i]):
            test_idx.append(i)
        else:
            train_idx.append(i)
    train_trace_df = copy.deepcopy(concurrency_df.iloc[train_idx])
    eval_trace_df = concurrency_df.iloc[test_idx]
    eval_trace_df = copy.deepcopy(
        eval_trace_df[eval_trace_df["num_concurrent_queries"] > 0]
    )
    return train_trace_df, eval_trace_df
