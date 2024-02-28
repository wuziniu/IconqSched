import numpy as np
from sklearn.linear_model import LinearRegression
from models.concurrency.base_model import ConcurPredictor


class SimpleLinearReg(ConcurPredictor):
    """
    Simple linear regression model for runtime prediction with concurrency
    simple linear regression considers runtime = (b + k * num_concurrency) * isolated_runtime
    Use linear regression to find b, k
    """

    def __init__(self):
        super().__init__()
        self.isolated_rt_cache = dict()
        self.use_train = True
        self.intercept_global = 0
        self.intercept = dict()
        self.slope_global = 0
        self.slope = dict()

    def train(self, trace_df, use_train=True, isolated_trace_df=None):
        self.get_isolated_runtime_cache(trace_df, isolated_trace_df)
        self.use_train = use_train
        concurrent_df = trace_df[trace_df["num_concurrent_queries"] > 0]

        global_y = []
        global_x = []
        for i, rows in concurrent_df.groupby("query_idx"):
            if i not in self.isolated_rt_cache:
                continue
            isolated_rt = self.isolated_rt_cache[i]
            concurrent_rt = rows["runtime"].values
            if use_train:
                num_concurrency = rows["num_concurrent_queries_train"].values
            else:
                num_concurrency = rows["num_concurrent_queries"].values
            global_y.append(concurrent_rt / isolated_rt)
            global_x.append(num_concurrency)
            model = LinearRegression()
            model.fit(num_concurrency.reshape(-1, 1), concurrent_rt / isolated_rt)
            self.intercept[i] = model.intercept_
            self.slope[i] = model.coef_
        global_y = np.concatenate(global_y)
        global_x = np.concatenate(global_x).reshape(-1, 1)
        model = LinearRegression()
        model.fit(global_x, global_y)
        self.intercept_global = model.intercept_
        self.slope_global = model.coef_[0]

    def predict(self, eval_trace_df, use_global=False):
        predictions = dict()
        labels = dict()
        for i, rows in eval_trace_df.groupby("query_idx"):
            if i not in self.isolated_rt_cache:
                continue
            isolated_rt = self.isolated_rt_cache[i]
            label = rows["runtime"].values
            labels[i] = label
            if self.use_train:
                num_concurrency = rows["num_concurrent_queries_train"].values
            else:
                num_concurrency = rows["num_concurrent_queries"].values
            if use_global:
                pred = (
                    num_concurrency * self.slope_global + self.intercept_global
                ) * isolated_rt
            else:
                pred = (
                    num_concurrency * self.slope[i] + self.intercept[i]
                ) * isolated_rt
            pred = np.maximum(pred, 0.001)
            predictions[i] = pred
        return predictions, labels
