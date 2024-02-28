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
        self.clustering = None
        self.isolated_rt_cache = dict()
        self.xgboost = None
        self.use_train = True
        self.use_pre_exec_info = False
        self.k = k

    def train(
        self, trace_df, use_train=True, isolated_trace_df=None, use_pre_exec_info=False
    ):
        self.use_train = use_train
        self.use_pre_exec_info = use_pre_exec_info
        self.get_isolated_runtime_cache(trace_df, isolated_trace_df)
        concurrent_df = trace_df[trace_df["num_concurrent_queries"] > 0]

        global_y = []
        global_x = []
        for i, rows in concurrent_df.groupby("query_idx"):
            if i not in self.isolated_rt_cache or len(rows) < 2:
                continue
            if use_train:
                concur_info = rows["concur_info_train"].values
            else:
                concur_info = rows["concur_info"].values
            global_y.append(rows["runtime"].values)
            query_feature = np.zeros((len(rows), self.k))
            query_feature[:, i] = self.isolated_rt_cache[i]
            # query_feature[:, i] = 1
            concur_query_feature = np.zeros((len(rows), self.k))
            for j in range(len(rows)):
                for c in concur_info[j]:
                    if c[0] in self.isolated_rt_cache:
                        concur_query_feature[j, c[0]] += self.isolated_rt_cache[c[0]]
                        # concur_query_feature[j, c[0]] += 1
                    else:
                        concur_query_feature[j, c[0]] += 1
            x = np.concatenate((query_feature, concur_query_feature), axis=1)
            if use_pre_exec_info:
                pre_exec_query_feature = np.zeros((len(rows), self.k))
                pre_exec_info = rows["pre_exec_info"].values
                for j in range(len(rows)):
                    for c in pre_exec_info[j]:
                        if c[0] in self.isolated_rt_cache:
                            pre_exec_query_feature[j, c[0]] += self.isolated_rt_cache[
                                c[0]
                            ]
                            # concur_query_feature[j, c[0]] += 1
                        else:
                            pre_exec_query_feature[j, c[0]] += 1
                x = np.concatenate((x, pre_exec_query_feature), axis=1)
            global_x.append(x)
        global_y = np.concatenate(global_y)
        global_x = np.concatenate(global_x)
        model = XGBRegressor(
            n_estimators=1000,
            max_depth=8,
            eta=0.2,
            subsample=1.0,
            eval_metric="mae",
            early_stopping_rounds=100,
        )
        train_idx = np.random.choice(
            len(global_y), size=int(0.8 * len(global_y)), replace=False
        )
        val_idx = [i for i in range(len(global_y)) if i not in train_idx]
        model.fit(
            global_x[train_idx],
            global_y[train_idx],
            eval_set=[(global_x[val_idx], global_y[val_idx])],
            verbose=False,
        )
        self.xgboost = model

    def predict(self, eval_trace_df, use_global=True):
        predictions = dict()
        labels = dict()
        for i, rows in eval_trace_df.groupby("query_idx"):
            if i not in self.isolated_rt_cache or len(rows) < 2:
                continue
            label = rows["runtime"].values
            labels[i] = label
            if self.use_train:
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
            if self.use_pre_exec_info:
                pre_exec_query_feature = np.zeros((len(rows), self.k))
                pre_exec_info = rows["pre_exec_info"].values
                for j in range(len(rows)):
                    for c in pre_exec_info[j]:
                        if c[0] in self.isolated_rt_cache:
                            pre_exec_query_feature[j, c[0]] += self.isolated_rt_cache[
                                c[0]
                            ]
                            # concur_query_feature[j, c[0]] += 1
                        else:
                            pre_exec_query_feature[j, c[0]] += 1
                x = np.concatenate((x, pre_exec_query_feature), axis=1)
            # if i == 0:
            #   for k in range(len(label)):
            #      print(x[k], label[k])

            pred = self.xgboost.predict(x)
            pred = np.maximum(pred, 0.001)
            predictions[i] = pred
        return predictions, labels
