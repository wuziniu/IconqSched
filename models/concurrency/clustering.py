import os
import numpy as np
from parser.utils import load_json
from parser.simple_parser import get_table_feature


class BaseCluster:
    def __init__(self, num_cluster):
        self.num_cluster = num_cluster
        self.query_runtime_cache = dict()

    def ingest_data(self, trace_df):
        for i, rows in trace_df.groupby("query_idx"):
            self.query_runtime_cache[i] = (np.median(rows["runtime"]), np.std(rows["runtime"]))


class KMeansCluster(BaseCluster):
    """
    Simple parser for sql, only parse the accessed table and selectivity
    """

    def __init__(self, num_cluster, schema_path, parser):
        super().__init__(num_cluster)
        assert os.path.exists(schema_path), f"Could not find schema.json ({schema_path})"
        self.schema = load_json(schema_path)
        self.all_table_list = self.schema["tables"]
        self.all_relation_list = self.schema["relationships"]
        self.all_column_list = []
        self.parser = parser



    def get_simple_feature(self, sql, pred_runtime):
        feature = get_table_feature(sql, self.all_table_list)
        return np.concatenate((feature, [pred_runtime]))
