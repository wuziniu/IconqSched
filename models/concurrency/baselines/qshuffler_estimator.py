import copy
import pandas as pd
import numpy as np
from typing import MutableMapping, Tuple, Mapping, Optional
from models.single.stage import SingleStage
from sklearn.linear_model import LinearRegression
from models.concurrency.base_model import ConcurPredictor
from models.single.query_clustering import KMeansCluster


class QEstimator(ConcurPredictor):
    def __init__(
        self,
        stage_model: SingleStage,
        cluster_model: Optional[KMeansCluster]=None,
        num_clusters: int = 10,
        mpl: int = 5,
    ):
        super().__init__()
        self.stage_model = stage_model
        self.cluster_model = cluster_model
        if cluster_model is not None:
            self.num_clusters = self.cluster_model.num_clusters
        else:
            self.num_clusters = num_clusters
        self.linear_models: MutableMapping[int, LinearRegression] = dict()
        self.isolated_rt: MutableMapping[int, float] = dict()
        self.mpl = mpl
        self.cost_threshold = None

    def train(self, trace_df: pd.DataFrame, isolated_trace_df: pd.DataFrame):
        self.get_isolated_runtime_cache(trace_df, isolated_trace_df)
        if self.cluster_model is None:
            self.cluster_model = KMeansCluster(self.stage_model, self.num_clusters)
            self.cluster_model.train()
        all_data = dict()
        all_label = dict()
        isolated_runtime_number = dict()
        isolated_runtime_sum = dict()
        represented_mix = dict()
        for i, rows in trace_df.groupby("query_idx"):
            i = int(i)
            cluster = self.cluster_model.infer(i)
            if cluster not in isolated_runtime_number:
                all_data[cluster] = []
                all_label[cluster] = []
                isolated_runtime_number[cluster] = 0
                isolated_runtime_sum[cluster] = 0
            if "runtime" in rows.columns:
                label = rows["runtime"].values
            else:
                label = rows["run_time_s"].values
            all_label[cluster].extend(list(label))
            isolated_runtime_number[cluster] += len(rows)
            isolated_runtime_sum[cluster] += np.sum(label)
            concur_info_full = rows["concur_info"].values
            for j in range(len(rows)):
                feature = np.zeros(self.cluster_model.num_clusters)
                feature[cluster] += 1
                for c in concur_info_full[j]:
                    query_idx = int(c[0])
                    query_cluster = self.cluster_model.infer(query_idx)
                    feature[query_cluster] += 1
                all_data[cluster].append(feature)
                feature_tuple = tuple(feature)
                if feature_tuple not in represented_mix:
                    represented_mix[feature_tuple] = dict()
                if cluster not in represented_mix[feature_tuple]:
                    represented_mix[feature_tuple][cluster] = [0, 0]
                represented_mix[feature_tuple][cluster][0] += 1
                represented_mix[feature_tuple][cluster][1] += label[j]

        for cluster in all_data:
            x = np.stack(all_data[cluster])
            y = np.asarray(all_label[cluster])
            assert x.shape[0] == len(y), f"feature and label size mismatch for cluster"
            linear_model = LinearRegression()
            _ = linear_model.fit(x, y)
            self.linear_models[cluster] = linear_model
            self.isolated_rt[cluster] = (
                isolated_runtime_sum[cluster] / isolated_runtime_number[cluster]
            )

        total_nro = 0
        total_runtime = 0
        for mix in represented_mix:
            mix_rt = represented_mix[mix]
            nro = 0
            sum_rt = 0
            for c in mix_rt:
                sum_rt += mix_rt[c][1]
                nro += mix_rt[c][1] / self.isolated_rt[c]
            nro = nro / self.mpl / self.mpl
            total_nro += sum_rt * nro
            total_runtime += sum_rt
        print(total_nro, total_runtime)
        self.cost_threshold = total_nro / total_runtime

    def predict(
        self, eval_trace_df: pd.DataFrame, use_global: bool = True
    ) -> Tuple[Mapping[int, np.ndarray], Mapping[int, np.ndarray]]:
        predictions = dict()
        labels = dict()
        for i, rows in eval_trace_df.groupby("query_idx"):
            i = int(i)
            cluster = self.cluster_model.infer(i)
            if "runtime" in rows.columns:
                label = rows["runtime"].values
            else:
                label = rows["run_time_s"].values
            labels[i] = label

            x = []
            concur_info_full = rows["concur_info"].values
            for j in range(len(rows)):
                feature = np.zeros(self.cluster_model.num_clusters)
                feature[cluster] += 1
                for c in concur_info_full[j]:
                    query_idx = int(c[0])
                    query_cluster = self.cluster_model.infer(query_idx)
                    feature[query_cluster] += 1
                x.append(feature)
            x = np.stack(x)
            pred = self.linear_models[cluster].predict(x)
            predictions[i] = pred
        return predictions, labels

    def online_inference(self, feature: np.ndarray) -> np.ndarray:
        nros = []
        for cluster in range(self.num_clusters):
            temp_feature = copy.deepcopy(feature)
            temp_feature[cluster] += 1
            nro = 0
            for j in self.linear_models:
                pred_j = self.linear_models[j].predict(temp_feature.reshape(1, -1))[0]
                nro += temp_feature[j] * pred_j / self.isolated_rt[j]
            nro = nro / self.mpl / self.mpl
            nros.append(nro)
        return np.asarray(nros)

    def get_query_type(self, query_idx: int):
        return self.cluster_model.infer(query_idx)
