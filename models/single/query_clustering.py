from sklearn.cluster import KMeans
import numpy as np
import pandas as pd
from models.single.stage import SingleStage
from typing import Optional


class KMeansCluster:
    def __init__(self, stage_model: SingleStage, num_clusters: int = 10):
        self.stage_model = stage_model
        self.num_clusters = num_clusters
        self.model = KMeans(self.num_clusters)
        self.clusters = None

    def train(self, trace_df: pd.DataFrame):
        # The current implement assumes the stage model has already observed the query pool,
        # will be easy to change in future
        all_feature = []
        if trace_df["query_idx"].nunique() <= self.num_clusters:
            self.clusters = dict()
            cluster_no = 0
            for i, rows in trace_df.groupby("query_idx"):
                i = int(i)
                self.clusters[i] = cluster_no
                cluster_no += 1
            self.num_clusters = len(self.clusters)
        else:
            for i in self.stage_model.all_feature:
                feature = self.stage_model.featurize_online(i)
                all_feature.append(feature)
            all_feature = np.stack(all_feature)
            self.clusters = self.model.fit_predict(all_feature)

    def infer(self, query_idx: int):
        if query_idx <= len(self.clusters):
            return self.clusters[query_idx]
        else:
            feature = self.stage_model.all_feature[query_idx]
            feature = feature.reshape(1, -1)
            pred = self.model.predict(feature)[0]
            return pred
