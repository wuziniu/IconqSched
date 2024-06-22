import copy
import torch
import pandas as pd
import numpy as np
from typing import Optional, Mapping, Tuple, List, Union
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence
from models.single.stage import SingleStage


def collate_fn_padding(batch):
    # Sort batch by sequence length (optional but recommended for efficiency)
    (feature, label, pre_info_length, query_idx) = zip(*batch)
    seq_lengths = [len(x) for x in feature]
    sort_idx = list(np.argsort(seq_lengths)[::-1])
    feature = [feature[i] for i in sort_idx]
    label = torch.tensor(label, dtype=torch.float)
    label = label[sort_idx]
    pre_info_length = torch.tensor(pre_info_length, dtype=torch.long)[sort_idx]
    query_idx = torch.tensor(query_idx, dtype=torch.long)[sort_idx]
    seq_lengths = torch.tensor(seq_lengths, dtype=torch.long)[sort_idx]
    # Pad sequences to the maximum length per batch
    padded_feature = pad_sequence(feature, batch_first=True, padding_value=0)
    return padded_feature, seq_lengths, label, pre_info_length, query_idx


def collate_fn_padding_preserve_order(batch):
    (feature, label, pre_info_length, query_idx) = zip(*batch)
    seq_lengths = [len(x) for x in feature]
    label = torch.tensor(label, dtype=torch.float)
    pre_info_length = torch.tensor(pre_info_length, dtype=torch.long)
    query_idx = torch.tensor(query_idx, dtype=torch.long)
    seq_lengths = torch.tensor(seq_lengths, dtype=torch.long)
    # Pad sequences to the maximum length per batch
    padded_feature = pad_sequence(feature, batch_first=True, padding_value=0)
    return padded_feature, seq_lengths, label, pre_info_length, query_idx


def collate_fn_padding_transformer(batch):
    (feature, label, pre_info_length, query_idx) = zip(*batch)
    seq_lengths = [len(x) for x in feature]
    sort_idx = list(np.argsort(seq_lengths)[::-1])
    feature = [feature[i] for i in sort_idx]
    label = torch.tensor(label, dtype=torch.float)
    label = label[sort_idx]
    pre_info_length = torch.tensor(pre_info_length, dtype=torch.long)[sort_idx]
    query_idx = torch.tensor(query_idx, dtype=torch.long)[sort_idx]
    # Pad sequences to the maximum length per batch
    padded_feature = pad_sequence(feature, batch_first=True, padding_value=0)
    seq_lengths = [seq_lengths[i] for i in sort_idx]
    src_key_padding_mask = torch.zeros(
        (padded_feature.shape[0], padded_feature.shape[1]), dtype=int
    )
    for i, seq_len in enumerate(seq_lengths):
        src_key_padding_mask[i, :seq_len] = 1
    return padded_feature, src_key_padding_mask, label, pre_info_length, query_idx


class QueryFeatureSeparatedDataset(Dataset):
    def __init__(self, feature, label, pre_info_length, query_idx):
        self.feature = feature
        self.label = label
        self.pre_info_length = pre_info_length
        self.query_idx = query_idx

    def __len__(self):
        return len(self.label)

    def __getitem__(self, idx):
        return (
            self.feature[idx],
            self.label[idx],
            self.pre_info_length[idx],
            self.query_idx[idx],
        )


def featurize_queries_complex_online(
    existing_query_features: List[np.ndarray],
    existing_query_concur_features: List[Optional[torch.Tensor]],
    existing_pre_info_length: List[int],
    queued_query_features: List[np.ndarray],
    existing_start_time: List[float],
    current_time: float,
    next_finish_idx_list: Optional[Union[int, List[int]]] = None,
    next_finish_time_list: Optional[Union[float, List[float]]] = None,
    get_next_finish: bool = False,
    get_next_finish_running_performance: bool = False,
    use_pre_exec_info: bool = False,
) -> Tuple[List[torch.Tensor], torch.Tensor]:
    # TODO: need to change
    global_x = []
    global_pre_info_length = []
    if next_finish_idx_list is not None and type(next_finish_idx_list) != list:
        next_finish_idx_list = [next_finish_idx_list]
        next_finish_time_list = [next_finish_time_list]
    for query_feature in queued_query_features:
        l_feature = len(query_feature)
        x = []
        # concurrent feature for the current queued query
        if get_next_finish and next_finish_idx_list is not None:
            # the feature of the current query when the next running query finishes running
            next_finish_x = [[] for _ in range(len(next_finish_idx_list))]
        else:
            next_finish_x = None
        for i, exist_q in enumerate(existing_query_features):
            concur_query_feature = np.zeros(l_feature * 2 + 5)
            concur_query_feature[:l_feature] = query_feature
            concur_query_feature[(l_feature + 2): (2 * l_feature + 2)] = exist_q
            concur_query_feature[l_feature] = 1
            concur_query_feature[2 * l_feature + 2] = (
                existing_start_time[i] - current_time
            )
            x.append(torch.FloatTensor(concur_query_feature))
            if next_finish_x is not None:
                for j in range(len(next_finish_idx_list)):
                    next_finish_idx = next_finish_idx_list[j]
                    next_finish_time = next_finish_time_list[j]
                    if i not in next_finish_idx_list[: (j+1)]:
                        unfinished_concur_query_feature = copy.deepcopy(
                            concur_query_feature
                        )
                        unfinished_concur_query_feature[2 * l_feature + 2] = (
                                existing_start_time[i] - next_finish_time
                        )
                        next_finish_x[j].append(torch.FloatTensor(unfinished_concur_query_feature))
        concur_query_feature = np.zeros(l_feature * 2 + 5)
        concur_query_feature[:l_feature] = query_feature
        x.append(torch.FloatTensor(concur_query_feature))
        global_pre_info_length.append(len(x))
        global_x.append(torch.stack(x))

        if next_finish_x is not None:
            for curr_next_finish_x in next_finish_x:
                if len(curr_next_finish_x) == 0:
                    # This can happen when the next finish query is the only query running in the system
                    concur_query_feature = np.zeros(l_feature * 2 + 5)
                    concur_query_feature[:l_feature] = query_feature
                    curr_next_finish_x.append(torch.FloatTensor(concur_query_feature))
                global_pre_info_length.append(len(curr_next_finish_x))
                global_x.append(torch.stack(curr_next_finish_x))

        # concurrent features for all existing (running queries) when this queued query is submitted
        for i in range(len(existing_query_concur_features)):
            global_pre_info_length.append(existing_pre_info_length[i])
            concur_query_feature = torch.zeros(l_feature * 2 + 5, dtype=torch.float)
            concur_query_feature[:l_feature] = torch.FloatTensor(
                existing_query_features[i]
            )
            concur_query_feature[(l_feature + 2): (2 * l_feature + 2)] = (
                torch.FloatTensor(query_feature)
            )
            concur_query_feature[l_feature + 1] = 1.0
            concur_query_feature[2 * l_feature + 2] = (
                existing_start_time[i] - current_time
            )
            if existing_query_concur_features[i] is None:
                x = concur_query_feature.reshape(1, -1)
            else:
                x = torch.clone(existing_query_concur_features[i])
                x = torch.cat((x, concur_query_feature.reshape(1, -1)), dim=0)
            global_x.append(x)
            if get_next_finish_running_performance and next_finish_idx_list is not None and len(next_finish_idx_list) != 0:
                finished_query_features = []
                for j in range(len(next_finish_idx_list)):
                    next_finish_idx = next_finish_idx_list[j]
                    next_finish_time = next_finish_time_list[j]
                    finished_query_features.append(existing_query_features[next_finish_idx])
                    if i not in next_finish_idx_list[: (j + 1)]:
                        concur_query_feature[2 * l_feature + 2] = (
                                existing_start_time[i] - next_finish_time
                        )
                        if existing_query_concur_features[i] is None:
                            x = concur_query_feature.reshape(1, -1)
                        else:
                            x = []
                            for k in range(len(existing_query_concur_features[i])):
                                k_query_feature = existing_query_concur_features[i][j][
                                                  (l_feature + 2): (2 * l_feature + 2)
                                                  ]
                                is_finished = False
                                for finished_query_feature in finished_query_features:
                                    if (
                                            not torch.sum(
                                                torch.abs(k_query_feature - finished_query_feature)
                                            )
                                                <= 1e-4
                                    ):
                                        # remove the finished query from its concurrent feature
                                        is_finished = True
                                        break
                                if not is_finished:
                                    x.append(existing_query_concur_features[i][j])
                            x.append(torch.FloatTensor(concur_query_feature))
                            x = torch.stack(x)
                    else:
                        # this query is already finished
                        x = torch.zeros((1, len(concur_query_feature)))
                    global_pre_info_length.append(max(len(x) - 1, 1))
                    global_x.append(x)
    global_pre_info_length = torch.LongTensor(global_pre_info_length)
    return global_x, global_pre_info_length


def featurize_queries_complex(
    concurrent_df: pd.DataFrame,
    predictions: np.ndarray,
    single_query_features: Mapping[int, np.ndarray],
    include_exit: bool = False,
    preserve_order: bool = False,
    use_pre_exec_info: bool = False,
    stagemodel: Optional[SingleStage] = None,
    ignore_short_running: bool = False,
    short_running_threshold: float = 5.0
) -> Tuple[List[torch.Tensor], torch.Tensor, torch.Tensor, torch.Tensor]:
    # Todo: hook predictions and query_features to call Stage model
    use_pre_exec_info = use_pre_exec_info & ("pre_exec_info" in concurrent_df.columns)
    global_y = []
    global_x = []
    global_pre_info_length = []
    global_query_idx = []
    query_order = []
    # reindexing the column to preserve the index
    concurrent_df["index"] = np.arange(len(concurrent_df))
    for i, rows in concurrent_df.groupby("query_idx"):
        i = int(i)
        if i not in predictions or i not in single_query_features:
            print(f"{i} not in predictions or {i} not in single_query_features")
            print(single_query_features.keys())
            assert False
            # if stagemodel is None or 'sql' not in concurrent_df.columns:
            #   continue
            # else:
            #   query_sql = rows['sql'].values[0]

        index_in_df = rows["index"].values
        query_order.append(index_in_df)
        concurrent_rt = rows["runtime"].values
        start_time = rows["start_time"].values
        global_y.append(concurrent_rt)
        n_rows = len(rows)
        query_feature = np.concatenate(
            (np.asarray([predictions[i]]), single_query_features[i])
        )
        l_feature = len(query_feature)
        if use_pre_exec_info:
            pre_exec_info = rows["pre_exec_info"].values
        else:
            pre_exec_info = [] * n_rows
        concur_info_train = rows["concur_info_train"].values
        concur_info_full = rows["concur_info"].values
        for j in range(n_rows):
            x = []
            current_pre_info_length = 0
            global_query_idx.append(i)

            for c in pre_exec_info[j] + concur_info_train[j]:
                if ignore_short_running:
                    c_runtime = c[2] - c[1]
                    if c_runtime < short_running_threshold:
                        continue
                concur_query_feature = np.zeros(l_feature * 2 + 5)
                concur_query_feature[:l_feature] = query_feature
                concur_query_feature[(l_feature + 2): (2 * l_feature + 2)] = (
                    np.concatenate(
                        (
                            np.asarray([predictions[c[0]]]),
                            single_query_features[c[0]],
                        )
                    )
                )
                current_pre_info_length += 1
                if c in pre_exec_info[j]:
                    concur_query_feature[2 * l_feature + 3] = 1
                    concur_query_feature[2 * l_feature + 2] = c[1] - start_time[j]
                else:
                    assert (
                        c[1] <= start_time[j]
                    ), f"parsing error in query index {i}, query number {j}"
                    concur_query_feature[l_feature] = 1
                    # encode the timestamp with a relative time in second, this is a negative value
                    # Todo: explore more timestamp embedding options
                    concur_query_feature[2 * l_feature + 2] = c[1] - start_time[j]
                x.append(torch.FloatTensor(concur_query_feature))

            current_pre_info_length += 1
            concur_query_feature = np.zeros(l_feature * 2 + 5)
            concur_query_feature[:l_feature] = query_feature
            x.append(torch.FloatTensor(concur_query_feature))

            for c in concur_info_full[j]:
                if c in concur_info_train[j]:
                    continue
                if ignore_short_running:
                    c_runtime = c[2] - c[1]
                    if c_runtime < short_running_threshold:
                        continue
                concur_query_feature = np.zeros(l_feature * 2 + 5)
                concur_query_feature[:l_feature] = query_feature
                concur_query_feature[(l_feature + 2): (2 * l_feature + 2)] = (
                    np.concatenate(
                        (
                            np.asarray([predictions[c[0]]]),
                            single_query_features[c[0]],
                        )
                    )
                )
                assert (
                    c[1] >= start_time[j]
                ), f"parsing error in query index {i}, query number {j}"
                concur_query_feature[l_feature + 1] = 1
                # this is a negative value
                concur_query_feature[2 * l_feature + 2] = start_time[j] - c[1]
                if include_exit:
                    # Todo: provide feature to indicate a query has left the instance? not helpful
                    end_time = rows["end_time"].values
                x.append(torch.FloatTensor(concur_query_feature))

            global_pre_info_length.append(current_pre_info_length)
            global_x.append(torch.stack(x))
    global_y = torch.FloatTensor(np.concatenate(global_y))
    global_pre_info_length = torch.LongTensor(global_pre_info_length)
    global_query_idx = torch.LongTensor(global_query_idx)
    query_order = np.concatenate(query_order)
    assert len(np.unique(query_order)) == len(query_order)
    if preserve_order:
        query_order = list(query_order)
        query_order_loc = []
        for i in range(len(query_order)):
            query_order_loc.append(query_order.index(i))
        global_x = [global_x[i] for i in query_order_loc]
        global_y = global_y[query_order_loc]
        global_pre_info_length = global_pre_info_length[query_order_loc]
        global_query_idx = global_query_idx[query_order_loc]
    return global_x, global_y, global_pre_info_length, global_query_idx
