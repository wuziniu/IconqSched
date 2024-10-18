import numpy as np
import pandas as pd
import collections
from typing import Optional, Union, Tuple
from models.single.cache import CachePredictor
from models.single.local_xgboost import SingleXGBoost
from models.feature.single_xgboost_feature import (
    find_top_k_operators,
    featurize_one_plan,
    get_top_k_table_by_size,
)
from parser.utils import load_json
from parser.parse_plan import parse_plan_online


class SingleStage:
    def __init__(
        self,
        capacity=5000,
        alpha=1.0,
        hash_bits=None,
        store_all=False,
        use_index=True,
        n_estimators=1000,
        max_depth=8,
        eta=0.2,
        eval_metric="mae",
        early_stopping_rounds=100,
        num_operators=20,
        use_size=True,
        use_log=True,
        true_card=False,
        use_table_features=True,
        use_table_selectivity=False,
        use_median=True,
        db_conn=None,
    ):
        self.cache = CachePredictor(
            capacity, alpha, hash_bits, store_all, use_index, use_median
        )
        self.local_model = SingleXGBoost(
            n_estimators, max_depth, eta, eval_metric, early_stopping_rounds
        )
        self.memory_est_cache = dict()
        self.num_operators = num_operators
        self.use_size = use_size
        self.use_log = use_log
        self.true_card = true_card
        self.operators = None
        self.use_table_features = use_table_features
        self.use_table_selectivity = use_table_selectivity
        self.all_feature = dict()
        self.all_table_size = None
        self.db_conn = db_conn
        self.column_id_mapping = dict()
        self.partial_column_name_mapping = collections.defaultdict(set)
        self.table_id_mapping = dict()

    def featurize_data(
        self,
        df: pd.DataFrame,
        parsed_queries_path: str,
        save_feature_file: Optional[str] = None,
    ):
        plans = load_json(parsed_queries_path, namespace=False)
        database_stats = plans["database_stats"]
        for i, column_stat in enumerate(database_stats["column_stats"]):
            table = column_stat["tablename"]
            column = column_stat["attname"]
            self.column_id_mapping[(table, column)] = i
            self.partial_column_name_mapping[column].add(table)

        for i, table_stat in enumerate(database_stats["table_stats"]):
            table = table_stat["relname"]
            self.table_id_mapping[table] = i

        self.operators = find_top_k_operators(plans=plans, k=self.num_operators)
        if self.use_table_features:
            self.all_table_size = get_top_k_table_by_size(plans=plans, k=15)
        self.all_feature = dict()
        for i in range(len(plans["parsed_plans"])):
            plan = plans["parsed_plans"][i]
            feature, memory_est = featurize_one_plan(
                plan,
                self.operators,
                self.all_table_size,
                use_size=self.use_size,
                use_log=self.use_log,
                true_card=self.true_card,
                return_memory_est=True,
            )
            self.memory_est_cache[i] = memory_est
            self.all_feature[i] = feature

        # Todo: should include data featurization for adhoc_queries, should be straight forward to add in
        features_df = []
        all_query_idx = df["query_idx"].values
        for i in range(len(df)):
            query_idx = all_query_idx[i]
            features_df.append(self.all_feature[query_idx])
        # for i, rows in df.groupby("query_idx"):
        #   feature = self.all_feature[i]
        #  row_idx = rows["index"].values
        # for j in row_idx:
        #    features_df[j] = feature
        df["features"] = features_df
        if save_feature_file is not None:
            df.to_csv(save_feature_file, header=True, index=False)
        return df

    def featurize_online(
        self, query_idx: int, query_sql: Optional[str] = None
    ) -> np.ndarray:
        if query_idx not in self.all_feature:
            plan = parse_plan_online(
                query_sql,
                self.column_id_mapping,
                self.partial_column_name_mapping,
                self.table_id_mapping,
                self.db_conn,
            )
            feature = featurize_one_plan(
                plan,
                self.operators,
                self.all_table_size,
                use_size=self.use_size,
                use_log=self.use_log,
                true_card=self.true_card,
            )
        else:
            feature = self.all_feature[query_idx]
        pred = self.cache.online_inference(query_idx, feature)
        if pred is None:
            pred = self.local_model.online_inference(feature)
        query_feature = np.concatenate((np.asarray([pred]), feature))
        return query_feature

    def train(self, df: pd.DataFrame) -> None:
        self.cache.ingest_data(df)
        self.local_model.train(df)

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        predictions, not_cached_idx = self.cache.predict(df)
        predictions = np.asarray(predictions)
        if len(not_cached_idx) != 0:
            predictions[not_cached_idx] = self.local_model.predict(
                df.iloc[not_cached_idx]
            )
        return predictions

    def evaluate(self, df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        predictions = self.predict(df)
        if "runtime" in df:
            labels = df["runtime"].values
        else:
            assert False
        abs_error = np.abs(predictions - labels)
        q_error = np.maximum(predictions / labels, labels / predictions)
        print(
            f"mean absolute error is {np.mean(abs_error)}, q-error is {np.mean(q_error)}"
        )
        for p in [50, 90, 95]:
            p_a = np.percentile(abs_error, p)
            p_q = np.percentile(q_error, p)
            print(f"{p}% absolute error is {p_a}, q-error is {p_q}")
        return predictions, labels
