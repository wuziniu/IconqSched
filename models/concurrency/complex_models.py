import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.optim as optim
from torch.nn.functional import l1_loss, mse_loss
from tqdm import tqdm
from models.concurrency.seq_to_seq import RNN, LSTM, TransformerModel
from models.feature.complex_rnn_features import (
    collate_fn_padding,
    collate_fn_padding_transformer,
    QueryFeatureSeparatedDataset,
    featurize_queries_complex,
)


def q_loss_func(
    input, target, min_val=0.001, small_val=5.0, penalty_negative=1e5, lambda_small=0.1
):
    """
    :param min_val: the minimal runtime you want the model to predict
    :param small_val: q_loss naturally favors small pred/label, put less weight on those values
    :return:
    """
    qerror = []
    for i in range(len(target)):
        # penalty for negative/too small estimates
        if (input[i] < min_val).data.numpy():
            # influence on loss for a negative estimate is >= penalty_negative constant
            q_err = (1 - input[i]) * penalty_negative
        # use l1_loss for small values, q_loss would explode
        elif (input[i] < small_val).data.numpy() and (
            target[i] < small_val
        ).data.numpy():
            q_err = torch.abs(target[i] - input[i]) * lambda_small
        # otherwise normal q error
        else:
            if (input[i] > target[i]).data.numpy():
                q_err = torch.log(input[i]) - torch.log(target[i])
            else:
                q_err = torch.log(target[i]) - torch.log(input[i])
        qerror.append(q_err)
    loss = torch.mean(torch.cat(qerror))
    return loss


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
        num_head=4,
        num_layers=4,
        batch_size=128,
        dropout=0.2,
        include_exit=False,
        last_output=True,
        rnn_type="lstm",
    ):
        self.stage_model = stage_model
        self.embedding_dim = embedding_dim
        self.hidden_size = hidden_size
        self.num_head = num_head
        self.num_layers = num_layers
        self.dropout = dropout
        self.include_exit = include_exit
        self.batch_size = batch_size
        self.rnn_type = rnn_type
        self.loss_func = None
        self.last_output = last_output
        if rnn_type == "vanilla":
            self.model = RNN(input_size, hidden_size, output_size, num_layers)
        elif rnn_type == "lstm":
            self.model = LSTM(
                input_size,
                embedding_dim,
                hidden_size,
                output_size,
                num_layers,
                dropout,
                last_output,
            )
        elif rnn_type == "transformer":
            self.model = TransformerModel(
                input_size,
                embedding_dim,
                num_head,
                hidden_size,
                num_layers,
                dropout,
                output_size,
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
        self.loss_func = loss_function
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

        val_x, val_y, val_pre_info_length, val_query_idx = featurize_queries_complex(
            val_df, predictions, single_query_features, include_exit=self.include_exit
        )
        (
            train_x,
            train_y,
            train_pre_info_length,
            train_query_idx,
        ) = featurize_queries_complex(
            train_df, predictions, single_query_features, include_exit=self.include_exit
        )

        train_dataset = QueryFeatureSeparatedDataset(
            train_x, train_y, train_pre_info_length, train_query_idx
        )
        if self.rnn_type == "transformer":
            collate_fn = collate_fn_padding_transformer()
        else:
            collate_fn = collate_fn_padding
        train_dataloader = DataLoader(
            train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            collate_fn=collate_fn,
        )
        val_dataset = QueryFeatureSeparatedDataset(
            val_x, val_y, val_pre_info_length, val_query_idx
        )
        val_dataloader = DataLoader(
            val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            collate_fn=collate_fn,
        )
        optimizer = optim.Adam(
            self.model.parameters(), lr=lr, weight_decay=weight_decay
        )
        for epoch in range(epochs):
            batch_loss = 0
            num_batch = 0
            self.model.train()
            for X, x_lengths, y, pre_info_length, query_idx in train_dataloader:
                optimizer.zero_grad()
                pred = self.model(X, x_lengths)
                y = y.reshape(-1, 1)
                if loss_function == "l1_loss":
                    loss = l1_loss(pred, y)
                elif loss_function == "mse_loss":
                    loss = mse_loss(pred, y)
                elif loss_function == "q_loss":
                    loss = q_loss_func(pred, y)
                else:
                    assert False, f"loss function {loss_function} is unrecognized"
                loss.backward()
                nn.utils.clip_grad_norm_(self.model.parameters(), 2)
                optimizer.step()
                batch_loss += loss.item()
                num_batch += 1
            if epoch % report_every == 0:
                train_loss = batch_loss / num_batch
                # Todo: implement eval loss
                print(
                    f"********Epoch {epoch}, training loss: {train_loss} || evaluation loss: ********"
                )
                _ = self.evaluate(val_dataloader, return_per_query=False)

    def predict(self, df, return_per_query=True):
        predictions = self.stage_model.cache.running_average
        single_query_features = dict()
        for i, f in enumerate(self.stage_model.all_feature):
            single_query_features[i] = f
        val_x, val_y, val_pre_info_length, val_query_idx = featurize_queries_complex(
            df, predictions, single_query_features, include_exit=self.include_exit
        )
        val_dataset = QueryFeatureSeparatedDataset(
            val_x, val_y, val_pre_info_length, val_query_idx
        )
        val_dataloader = DataLoader(
            val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            collate_fn=collate_fn_padding,
        )
        return self.evaluate(val_dataloader, return_per_query=return_per_query)

    def evaluate(self, val_dataloader, return_per_query=False):
        self.model.eval()
        all_pred = []
        all_label = []
        all_query_idx = []
        for X, x_lengths, y, pre_info_length, query_idx in tqdm(val_dataloader):
            pred = self.model(X, x_lengths)
            pred = pred.reshape(-1).detach().numpy()
            label = y.numpy()
            all_pred.append(pred)
            all_label.append(label)
            all_query_idx.append(query_idx.numpy())
        all_pred = np.concatenate(all_pred)
        all_pred = np.maximum(all_pred, 0.01)
        all_label = np.concatenate(all_label)
        all_query_idx = np.concatenate(all_query_idx)
        abs_error = np.abs(all_pred - all_label)
        q_error = np.maximum(all_pred / all_label, all_label / all_pred)
        for p in [50, 90, 95]:
            p_a = np.percentile(abs_error, p)
            p_q = np.percentile(q_error, p)
            print(f"{p}% absolute error is {p_a}, q-error is {p_q}")
        if return_per_query:
            preds_per_query = dict()
            labels_per_query = dict()
            for i in range(len(all_query_idx)):
                q_idx = int(all_query_idx[i])
                if q_idx not in preds_per_query:
                    preds_per_query[q_idx] = []
                    labels_per_query[q_idx] = []
                preds_per_query[q_idx].append(all_pred[i])
                labels_per_query[q_idx].append(all_label[i])
            return preds_per_query, labels_per_query
        return all_pred, all_label

    def save_model(self, directory):
        model_path = os.path.join(
            directory,
            f"{self.rnn_type}_{self.hidden_size}_{self.num_layers}_{self.loss_func}",
        )
        torch.save(self.model.state_dict(), model_path)

    def load_model(self, directory):
        model_path = os.path.join(
            directory,
            f"{self.rnn_type}_{self.hidden_size}_{self.num_layers}_{self.loss_func}",
        )
        self.model.load_state_dict(torch.load(model_path))
