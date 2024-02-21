import numpy as np
from models.concurrency.base_model import ConcurPredictor
from xgboost import XGBRegressor


class XGBoostPredictor(ConcurPredictor):
    """
    Consider k unique concurrent queries (or group into k classes)
    For each query represent it as k-dimensional one hot vector
         Then, identify its concurrent queries, represent them as a k-dimensional vector
         [rt_q1, rt_q2, ..., rt_qk], where rt_qi is the runtime of query i running in isolation
         Concatenate two features together to build simple XGboost model to predict the runtime
    """
    def __init__(self, k=100):
        super().__init__()
        self.clustering = None
        self.isolated_rt_cache = dict()
        self.xgboost = None
        self.k = k

    def train(self, trace_df, use_train=True, isolated_trace_df=None):
        self.get_isolated_runtime_cache(trace_df, isolated_trace_df)
        concurrent_df = trace_df[trace_df['num_concurrent_queries'] > 0]

        global_y = []
        global_x = []
        for i, rows in concurrent_df.groupby("query_idx"):
            if i not in self.isolated_rt_cache or len(rows) < 10:
                continue
            if use_train:
                concur_info = rows["concur_info_train"].values
            else:
                concur_info = rows["concur_info"].values
            global_y.append(rows["runtime"].values)
            query_feature = np.zeros((len(rows), self.k))
            query_feature[:, i] = self.isolated_rt_cache[i]

            concur_query_feature = np.zeros((len(rows), self.k))
            for j in range(len(rows)):
                for c in concur_info[j]:
                    if c[0] in self.isolated_rt_cache:
                        concur_query_feature[j, c[0]] += self.isolated_rt_cache[c[0]]
                    else:
                        concur_query_feature[j, c[0]] += 2
            x = np.concatenate((query_feature, concur_query_feature), axis=1)
            global_x.append(x)
        global_y = np.concatenate(global_y)
        global_x = np.concatenate(global_x)
        model = XGBRegressor(n_estimators=500, max_depth=7, eta=0.1, subsample=1.0,
                             eval_metric="rmse")
        train_idx = np.random.choice(len(global_y), size=int(0.8 * len(global_y)), replace=False)
        val_idx = [i for i in range(len(global_y)) if i not in train_idx]
        model.fit(global_x[train_idx], global_y[train_idx],
                  eval_set=[(global_x[val_idx], global_y[val_idx])],
                  early_stopping_rounds=100,
                  verbose=False
                 )
        self.xgboost = model


    def predict(self, eval_trace_df, use_global=True, use_train=True):
        predictions = dict()
        labels = dict()
        for i, rows in eval_trace_df.groupby("query_idx"):
            if i not in self.isolated_rt_cache or len(rows) < 10:
                continue
            label = rows["runtime"].values
            labels[i] = label
            if use_train:
                concur_info = rows["concur_info_train"].values
            else:
                concur_info = rows["concur_info"].values
            query_feature = np.zeros((len(rows), self.k))
            query_feature[:, i] = self.isolated_rt_cache[i]

            concur_query_feature = np.zeros((len(rows), self.k))
            for j in range(len(rows)):
                for c in concur_info[j]:
                    if c[0] in self.isolated_rt_cache:
                        concur_query_feature[j, c[0]] += self.isolated_rt_cache[c[0]]
                    else:
                        concur_query_feature[j, c[0]] += 2
            x = np.concatenate((query_feature, concur_query_feature), axis=1)
            pred = self.xgboost.predict(x)
            pred = np.maximum(pred, 0.001)
            predictions[i] = pred
        return predictions, labels

