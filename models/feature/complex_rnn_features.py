import torch
import numpy as np
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence


def collate_fn_padding(batch):
    # Sort batch by sequence length (optional but recommended for efficiency)
    (feature, label, pre_info_length) = zip(*batch)
    seq_lengths = [len(x) for x in feature]
    sort_idx = list(np.argsort(seq_lengths)[::-1])
    feature = [feature[i] for i in sort_idx]
    label = torch.tensor(label, dtype=torch.float)
    label = label[sort_idx]
    pre_info_length = torch.tensor(pre_info_length, dtype=torch.long)[sort_idx]
    seq_lengths = torch.tensor(seq_lengths, dtype=torch.long)[sort_idx]
    # Pad sequences to the maximum length per batch
    padded_feature = pad_sequence(feature, batch_first=True, padding_value=0)
    return padded_feature, seq_lengths, label, pre_info_length


class QueryFeatureSeparatedDataset(Dataset):
    def __init__(self, feature, label, pre_info_length):
        self.feature = feature
        self.label = label
        self.pre_info_length = pre_info_length

    def __len__(self):
        return len(self.label)

    def __getitem__(self, idx):
        return self.feature[idx], self.label[idx], self.pre_info_length[idx]


def featurize_queries_complex(
    concurrent_df, predictions, single_query_features, include_exit=False
):
    # Todo: hook predictions and query_features to call Stage model
    global_y = []
    global_x = []
    global_pre_info_length = []
    for i, rows in concurrent_df.groupby("query_idx"):
        if i not in predictions or i not in single_query_features:
            continue
        concurrent_rt = rows["runtime"].values
        start_time = rows["start_time"].values
        global_y.append(concurrent_rt)
        n_rows = len(rows)
        query_feature = np.concatenate(
            (np.asarray([predictions[i]]), single_query_features[i])
        )
        l_feature = len(query_feature)
        concur_info_train = rows["concur_info_train"].values
        concur_info_full = rows["concur_info"].values
        for j in range(n_rows):
            x = []
            global_pre_info_length.append(len(concur_info_train[j]))
            if len(concur_info_full[j]) == 0:
                concur_query_feature = np.zeros(l_feature * 2 + 5)
                concur_query_feature[:l_feature] = query_feature
                x.append(torch.FloatTensor(concur_query_feature))
            else:
                for c in concur_info_full[j]:
                    concur_query_feature = np.zeros(l_feature * 2 + 5)
                    concur_query_feature[:l_feature] = query_feature
                    concur_query_feature[
                        (l_feature + 2) : (2 * l_feature + 2)
                    ] = np.concatenate(
                        (np.asarray([predictions[c[0]]]), single_query_features[c[0]])
                    )
                    if c in concur_info_train[j]:
                        assert c[1] <= start_time[j]
                        concur_query_feature[l_feature] = 1
                        # encode the timestamp with a relative time in second, this is a negative value
                        # Todo: explore more timestamp embedding options
                        concur_query_feature[2 * l_feature + 2] = c[1] - start_time[j]
                    else:
                        assert c[1] >= start_time[j]
                        concur_query_feature[l_feature + 1] = 1
                        # this is a negative value
                        concur_query_feature[2 * l_feature + 2] = start_time[j] - c[1]
                    if include_exit:
                        # Todo: provide feature to indicate a query has left the instance
                        end_time = rows["end_time"].values
                    x.append(torch.FloatTensor(concur_query_feature))
            global_x.append(torch.stack(x))
    global_y = torch.FloatTensor(np.concatenate(global_y))
    global_pre_info_length = torch.LongTensor(global_pre_info_length)

    return global_x, global_y, global_pre_info_length
