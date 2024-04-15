import numpy as np
from typing import Optional


class CachePredictor:
    def __init__(
        self, capacity=5000, alpha=1.0, hash_bits=None, store_all=False, use_index=False
    ):
        self.all_query_hash_val = []
        self.running_average = dict()
        self.most_recent = dict()
        self.running_std = dict()
        self.num_obervation = dict()
        self.all_rt = dict()
        self.alpha = alpha
        self.capacity = capacity
        self.store_all = store_all
        self.hash_bits = hash_bits
        self.use_index = use_index

    def hash_feature(self, feature: np.ndarray) -> int:
        hash_val = hash(tuple(list(feature)))
        if self.hash_bits is not None:
            hash_val = hash_val % (2**self.hash_bits)
        return hash_val

    def ingest_data(self, df):
        for i, rows in df.groupby("query_idx"):
            if self.use_index:
                hash_val = i
            else:
                hash_val = self.hash_feature(rows["feature"].iloc[0])
            if hash_val not in self.running_average:
                self.all_query_hash_val.append(hash_val)
                self.running_average[hash_val] = np.average(rows["runtime"])
                self.most_recent[hash_val] = rows["runtime"].values[-1]
                self.running_std[hash_val] = np.std(rows["runtime"])
                self.num_obervation[hash_val] = len(rows)
                if self.store_all:
                    self.all_rt[hash_val] = list(rows["runtime"])
            else:
                observed_rows = self.num_obervation[hash_val]
                num_rows = len(rows)
                new_rows = observed_rows + num_rows
                self.running_average[hash_val] = (
                    observed_rows * self.running_average[hash_val]
                    + new_rows * np.average(rows["runtime"])
                ) / new_rows
                self.most_recent[hash_val] = rows["runtime"].values[-1]
                # Todo: this is not the correct formula
                self.running_std[hash_val] = (
                    observed_rows * self.running_std[hash_val]
                    + new_rows * np.std(rows["runtime"])
                ) / new_rows
                self.num_obervation[hash_val] = new_rows
                if self.store_all:
                    self.all_rt[hash_val].extend(list(rows["runtime"]))
            if len(self.all_query_hash_val) > self.capacity:
                # Todo: implement eviction
                continue

    def predict(self, df):
        predictions = []
        not_cached_idx = []
        for i in range(len(df)):
            if self.use_index:
                hash_val = df["query_idx"].iloc[i]
            else:
                hash_val = self.hash_feature(df["feature"].iloc[i])
            if hash_val not in self.running_average:
                predictions.append(-1)
                not_cached_idx.append(i)
            else:
                pred = self.running_average[hash_val] * self.alpha + self.most_recent[
                    hash_val
                ] * (1 - self.alpha)
                predictions.append(pred)
        return predictions, not_cached_idx

    def online_inference(self, query_idx: int, feature: np.ndarray) -> Optional[float]:
        if self.use_index:
            hash_val = query_idx
        else:
            hash_val = self.hash_feature(feature)
        if hash_val not in self.running_average:
            return None
        else:
            pred = self.running_average[hash_val] * self.alpha + self.most_recent[
                hash_val
            ] * (1 - self.alpha)
            return pred
