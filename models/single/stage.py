import numpy as np
from models.single.cache import CachePredictor
from models.single.local_xgboost import SingleXGBoost
from models.feature.single_xgboost_feature import (
    find_top_k_operators,
    featurize_one_plan,
    get_top_k_table_by_size,
)
from parser.utils import load_json


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
        use_table_features=False,
        use_table_selectivity=False,
    ):
        self.cache = CachePredictor(capacity, alpha, hash_bits, store_all, use_index)
        self.local_model = SingleXGBoost(
            n_estimators, max_depth, eta, eval_metric, early_stopping_rounds
        )
        self.num_operators = num_operators
        self.use_size = use_size
        self.use_log = use_log
        self.true_card = true_card
        self.operators = None
        self.use_table_features = use_table_features
        self.use_table_selectivity = use_table_selectivity
        self.all_feature = []
        self.all_table_size = None

    def featurize_data(self, df, parsed_queries_path, save_feature_file=None):
        plans = load_json(parsed_queries_path, namespace=False)
        self.operators = find_top_k_operators(plans=plans, k=self.num_operators)
        if self.use_table_features:
            self.all_table_size = get_top_k_table_by_size(plans=plans, k=15)
        self.all_feature = []
        for i in range(len(plans["parsed_plans"])):
            plan = plans["parsed_plans"][i]
            feature = featurize_one_plan(
                plan,
                self.operators,
                self.all_table_size,
                use_size=self.use_size,
                use_log=self.use_log,
                true_card=self.true_card,
            )
            self.all_feature.append(feature)
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

    def train(self, df):
        self.cache.ingest_data(df)
        self.local_model.train(df)

    def predict(self, df):
        predictions, not_cached_idx = self.cache.predict(df)
        predictions = np.asarray(predictions)
        if len(not_cached_idx) != 0:
            predictions[not_cached_idx] = self.local_model.predict(
                df.iloc[not_cached_idx]
            )
        return predictions
