import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.optim as optim
from torch.nn.functional import l1_loss, mse_loss
from tqdm import tqdm
from typing import Union, Mapping, Tuple, List, Optional
from models.single.stage import SingleStage
from models.concurrency.seq_to_seq import RNN, LSTM, TransformerModel
from models.feature.complex_rnn_features import (
    collate_fn_padding,
    collate_fn_padding_preserve_order,
    collate_fn_padding_transformer,
    QueryFeatureSeparatedDataset,
    featurize_queries_complex,
    featurize_queries_complex_online,
)


def q_loss_func(
    input: torch.Tensor,
    target: torch.Tensor,
    min_val: float = 0.001,
    small_val: float = 5.0,
    penalty_negative: float = 1e5,
    lambda_small: float = 0.1,
):
    # loss function that minimizes q-error
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
    def __init__(self, input_dim: int, hidden_dim: int = 64, layers: int = 3):
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

    def forward(self, x1: torch.Tensor, x2: torch.Tensor, x3: torch.Tensor):
        y1 = self.model(x1, x2)
        if self.is_train:
            pred = self.model(y1, x3)
        else:
            pred = y1
        return torch.maximum(pred, torch.tensor(0.01))


class ConcurrentRNN:
    def __init__(
        self,
        stage_model: SingleStage,
        model_prefix: str,
        input_size: int,
        embedding_dim: int,
        hidden_size: int,
        output_size: int = 1,
        num_head: int = 4,
        num_layers: int = 4,
        batch_size: int = 128,
        dropout: float = 0.2,
        include_exit: bool = False,
        last_output: bool = True,
        loss_function: str = "q_loss",
        rnn_type: str = "lstm",
        use_separation: bool = False,
        use_pre_exec_info: bool = True,
        ignore_short_running: bool = False,
        short_running_threshold: float = 5.0,
    ):
        """
        :param stage_model: stage model for predicting and featurize one query
        :param model_prefix: the name for the trained model
        :param input_size: input feature dimension
        :param embedding_dim: feature embedding dimension for DNN
        :param hidden_size: hidden layer size for DNN
        :param output_size: output size, should always be 1
        :param num_head: number of heads for transformer
        :param num_layers: number of layers for DNN
        :param batch_size: batch size for training
        :param dropout: dropout percentage for DNN
        :param include_exit: If set to true (not recommended), it will provide a feature when a query finish
        :param last_output: use the last hidden state and output of LSTM to make final prediction.
                            If set to false, it will use the average instead of the last (not recommended)
        :param loss_function: choose among "q_loss" and "l1_loss" and "mse_loss"
        :param rnn_type: choose among "vanilla", "lstm", "transformer" (only recommend lstm)
        :param use_separation: explicitly separate the influence of queries submitted before and after target query
        :param use_pre_exec_info: adding queries that recently finished in the system
        :param ignore_short_running: set to true to directly submit short running query to avoid overhead
        :param shorting_running_threshold: consider query with predicted threshold to be shorting running query
        """
        self.model_prefix = model_prefix
        self.stage_model = stage_model
        self.embedding_dim = embedding_dim
        self.hidden_size = hidden_size
        self.num_head = num_head
        self.num_layers = num_layers
        self.dropout = dropout
        self.include_exit = include_exit
        self.batch_size = batch_size
        self.rnn_type = rnn_type
        self.loss_func = loss_function
        self.last_output = last_output
        self.use_seperation = use_separation
        self.use_pre_exec_info = use_pre_exec_info
        self.ignore_short_running = ignore_short_running
        self.short_running_threshold = short_running_threshold
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
                use_seperation=use_separation,
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
            assert False, f"unrecognized rnn type: {rnn_type}"

    def train(
        self,
        df,
        test_df=None,
        lr=0.001,
        weight_decay=2e-5,
        epochs=200,
        loss_function=None,
        report_every=5,
        val_on_test=False,
    ):
        if loss_function is not None:
            self.loss_func = loss_function
        predictions = self.stage_model.cache.running_average
        single_query_features = self.stage_model.all_feature
        # single_query_features = dict()
        # for i, f in enumerate(self.stage_model.all_feature):
        #   single_query_features[i] = f

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
            val_df,
            predictions,
            single_query_features,
            include_exit=self.include_exit,
            use_pre_exec_info=self.use_pre_exec_info,
            ignore_short_running=self.ignore_short_running,
            short_running_threshold=self.short_running_threshold
        )
        (
            train_x,
            train_y,
            train_pre_info_length,
            train_query_idx,
        ) = featurize_queries_complex(
            train_df,
            predictions,
            single_query_features,
            include_exit=self.include_exit,
            use_pre_exec_info=self.use_pre_exec_info,
            ignore_short_running=self.ignore_short_running,
            short_running_threshold=self.short_running_threshold
        )

        train_dataset = QueryFeatureSeparatedDataset(
            train_x, train_y, train_pre_info_length, train_query_idx
        )
        if self.rnn_type == "transformer":
            collate_fn = collate_fn_padding_transformer
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
                pred = self.model(X, x_lengths, pre_info_length)
                y = y.reshape(-1, 1)
                if self.loss_func == "l1_loss":
                    loss = l1_loss(pred, y)
                elif self.loss_func == "mse_loss":
                    loss = mse_loss(pred, y)
                elif self.loss_func == "q_loss":
                    loss = q_loss_func(pred, y)
                else:
                    assert False, f"loss function {self.loss_func} is unrecognized"
                loss.backward()
                nn.utils.clip_grad_norm_(self.model.parameters(), 2)
                optimizer.step()
                batch_loss += loss.item()
                num_batch += 1
            if epoch % report_every == 0:
                train_loss = batch_loss / num_batch
                print(
                    f"********Epoch {epoch}, training loss: {train_loss} || evaluation loss: ********"
                )
                _ = self.evaluate(val_dataloader, return_per_query=False)

    def predict(
        self,
        df: pd.DataFrame,
        return_per_query: bool = True,
        use_pre_info_only: bool = False,
    ) -> Union[
        Tuple[Mapping[int, list], Mapping[int, list]], Tuple[np.ndarray, np.ndarray]
    ]:
        predictions = self.stage_model.cache.running_average
        single_query_features = self.stage_model.all_feature
        val_x, val_y, val_pre_info_length, val_query_idx = featurize_queries_complex(
            df,
            predictions,
            single_query_features,
            include_exit=self.include_exit,
            preserve_order=True,
            use_pre_exec_info=self.use_pre_exec_info,
        )
        val_dataset = QueryFeatureSeparatedDataset(
            val_x, val_y, val_pre_info_length, val_query_idx
        )
        val_dataloader = DataLoader(
            val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            collate_fn=collate_fn_padding_preserve_order,
        )
        return self.evaluate(
            val_dataloader,
            return_per_query=return_per_query,
            use_pre_info_only=use_pre_info_only,
        )

    def evaluate(
        self,
        val_dataloader: DataLoader,
        return_per_query: bool = False,
        use_pre_info_only: bool = False,
    ) -> Union[
        Tuple[Mapping[int, list], Mapping[int, list]], Tuple[np.ndarray, np.ndarray]
    ]:
        self.model.eval()
        all_pred = []
        all_label = []
        all_query_idx = []
        for X, x_lengths, y, pre_info_length, query_idx in tqdm(val_dataloader):
            if use_pre_info_only:
                pred = self.model.model_forward_pre_info(X, pre_info_length)
            else:
                pred = self.model(X, x_lengths, pre_info_length)
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

    def online_inference(
        self,
        existing_query_features: List[np.ndarray],
        existing_query_concur_features: List[Optional[torch.Tensor]],
        existing_pre_info_length: List[int],
        queued_query_features: List[np.ndarray],
        existing_start_time: List[float],
        current_time: float,
        next_finish_idx: Optional[int] = None,
        next_finish_time: Optional[float] = None,
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

    def save_model(self, directory: str):
        sep = "w_sep" if self.use_seperation else "wo_sep"
        model_path = os.path.join(
            directory,
            f"{self.model_prefix}_{self.rnn_type}_{self.hidden_size}_{self.num_layers}_{self.loss_func}_{sep}",
        )
        torch.save(self.model.state_dict(), model_path)

    def load_model(self, directory: str):
        sep = "w_sep" if self.use_seperation else "wo_sep"
        model_path = os.path.join(
            directory,
            f""
            f"{self.model_prefix}_{self.rnn_type}_{self.hidden_size}_{self.num_layers}_{self.loss_func}_{sep}",
        )
        self.model.load_state_dict(torch.load(model_path))
