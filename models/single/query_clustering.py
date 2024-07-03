from sklearn.cluster import KMeans
import numpy as np
from models.single.stage import SingleStage


class KMeansCluster:
    def __init__(self, stage_model: SingleStage, num_clusters: int = 10):
        self.stage_model = stage_model
        self.num_clusters = num_clusters
        self.model = KMeans(self.num_clusters)
        self.clusters = None

    def train(self):
        # The current implement assumes the stage model has already observed the query pool,
        # will be easy to change in future
        all_feature = []
        for i in self.stage_model.all_feature:
            feature = self.stage_model.featurize_online(i)
            all_feature.append(feature)
        all_feature = np.stack(all_feature)
        self.clusters = self.model.fit_predict(all_feature)

    def infer(self, query_idx: int):
        return self.clusters[query_idx]
