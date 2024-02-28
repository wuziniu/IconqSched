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
            self.query_runtime_cache[i] = (
                np.median(rows["runtime"]),
                np.std(rows["runtime"]),
            )


class KMeansCluster(BaseCluster):
    """
    Simple parser for sql, only parse the accessed table and selectivity
    """

    def __init__(self, num_cluster, schema_path, query_file, parser=None):
        super().__init__(num_cluster)
        assert os.path.exists(
            schema_path
        ), f"Could not find schema.json ({schema_path})"
        assert os.path.exists(query_file), f"Could not find query file ({query_file})"
        self.schema = load_json(schema_path)
        self.all_table_list = self.schema["tables"]
        self.all_relation_list = self.schema["relationships"]
        self.all_column_list = []
        with open(query_file, "r") as f:
            self.all_queries = f.readline()
        self.parser = parser
        if parser is None:
            self.parser = get_table_feature

    def featurize_queries(self, trace_df):
        self.ingest_data(trace_df)
        query_idx = list(self.query_runtime_cache.keys())
        all_feature = []
        for i in query_idx:
            sql = self.all_queries[i]
            sql_feature = self.parser(sql, self.all_table_list)
            feature = np.concatenate((sql_feature, list(self.query_runtime_cache[i])))
            all_feature.append(feature)
