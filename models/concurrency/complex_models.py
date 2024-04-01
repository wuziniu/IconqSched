import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.optim as optim
from torch.nn.functional import l1_loss, mse_loss
from tqdm import tqdm
from models.concurrency.seq_to_seq import RNN, LSTM
from models.feature.complex_rnn_features import (
    collate_fn_padding,
    QueryFeatureSeparatedDataset,
    featurize_queries_complex,
)


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


class ConcurrentRNN:
    def __init__(
        self,
        stage_model,
        input_size,
        embedding_dim,
        hidden_size,
        output_size=1,
        num_layers=4,
        batch_size=128,
        include_exit=False,
        rnn_type="lstm",
    ):
        self.stage_model = stage_model
        self.embedding_dim = embedding_dim
        self.include_exit = include_exit
        self.batch_size = batch_size
        self.rnn_type = rnn_type
        if rnn_type == "vanilla":
            self.model = RNN(input_size, hidden_size, output_size, num_layers)
        elif rnn_type == "lstm":
            self.model = LSTM(
                input_size, embedding_dim, hidden_size, output_size, num_layers
            )
        else:
            # Todo: implement transformer
            assert False, f"unrecognized rnn type: {rnn_type}"

    def train(
        self,
        df,
        test_df=None,
        lr=0.001,
        weight_decay=2e-5,
        epochs=200,
        loss_function="l1_loss",
        report_every=5,
        val_on_test=False,
    ):
        predictions = self.stage_model.cache.running_average
        single_query_features = dict()
        for i, f in enumerate(self.stage_model.all_feature):
            single_query_features[i] = f

        if val_on_test:
            assert (
                test_df is not None
            ), "must provide test dataframe to evaluate on test"
            val_df = test_df
            train_df = df
        else:
            # random train-eval split
            train_idx = np.random.choice(
                len(df), size=int(0.85 * len(df)), replace=False
            )
            val_idx = [i for i in range(len(df)) if i not in train_idx]
            val_df = df.iloc[val_idx]
            train_df = df.iloc[train_idx]

        val_x, val_y, val_pre_info_length = featurize_queries_complex(
            val_df, predictions, single_query_features, include_exit=self.include_exit
        )
        train_x, train_y, train_pre_info_length = featurize_queries_complex(
            train_df, predictions, single_query_features, include_exit=self.include_exit
        )

        train_dataset = QueryFeatureSeparatedDataset(
            train_x, train_y, train_pre_info_length
        )
        train_dataloader = DataLoader(
            train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            collate_fn=collate_fn_padding,
        )
        val_dataset = QueryFeatureSeparatedDataset(val_x, val_y, val_pre_info_length)
        val_dataloader = DataLoader(
            val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            collate_fn=collate_fn_padding,
        )
        optimizer = optim.Adam(
            self.model.parameters(), lr=lr, weight_decay=weight_decay
        )
        for epoch in range(epochs):
            batch_loss = 0
            num_batch = 0
            self.model.train()
            for X, x_lengths, y, pre_info_length in train_dataloader:
                optimizer.zero_grad()
                pred = self.model(X, x_lengths)
                pred = pred.reshape(-1)
                if loss_function == "l1_loss":
                    loss = l1_loss(pred, y)
                elif loss_function == "mse_loss":
                    loss = mse_loss(pred, y)
                else:
                    assert False, f"loss function {loss_function} is unrecognized"
                loss.backward()
                optimizer.step()
                batch_loss += loss.item()
                num_batch += 1
            if epoch % report_every == 0:
                train_loss = batch_loss / num_batch
                # Todo: implement eval loss
                print(
                    f"********Epoch {epoch}, training loss: {train_loss} || evaluation loss: ********"
                )
                self.evaluate(val_dataloader)

    def predict(self, df):
        return

    def evaluate(self, val_dataloader):
        self.model.eval()
        all_pred = []
        all_label = []
        for X, x_lengths, y, pre_info_length in tqdm(val_dataloader):
            pred = self.model(X, x_lengths)
            pred = pred.reshape(-1).detach().numpy()
            label = y.numpy()
            all_pred.append(pred)
            all_label.append(label)
        all_pred = np.concatenate(all_pred)
        all_pred = np.maximum(all_pred, 0.01)
        all_label = np.concatenate(all_label)
        abs_error = np.abs(all_pred - all_label)
        q_error = np.maximum(all_pred / all_label, all_label / all_pred)
        for p in [50, 90, 95]:
            p_a = np.percentile(abs_error, p)
            p_q = np.percentile(q_error, p)
            print(f"{p}% absolute error is {p_a}, q-error is {p_q}")
