import numpy as np
from sklearn.linear_model import LinearRegression
import scipy.optimize as optimization
from models.concurrency.base_model import ConcurPredictor


def queueing_func(x, a1, a2, b1):
    """
    a1 represents the average exec-time of a random query under concurrency
    b1 represents the max level of concurrency in a system
    a2 represents the average impact on a query's runtime when executed concurrently with other queries
    """
    num_concurrency, isolated_runtime = x
    return (a1 * np.maximum(num_concurrency - b1, 0)) + (1 + a2 * np.minimum(num_concurrency, b1)) * isolated_runtime


class SimpleFitCurve(ConcurPredictor):
    """
    Simple fit curve model for runtime prediction with concurrency
    runtime = queue_time(num_concurrency) + alpha(num_concurrency) * isolated_runtime
            = (a1 * max(num_concurrency-b1, 0)) + (1 + a2*min(num_concurrency, b1)) * isolated_runtime
    optimize a1, b1, b2
    """
    def __init__(self):
        super().__init__()
        self.isolated_rt_cache = dict()
        self.a1_global = 0
        self.a1 = dict()
        self.b1_global = 0
        self.b1 = dict()
        self.a2_global = 0
        self.a2 = dict()

    def train(self, trace_df, use_train=True, isolated_trace_df=None):
        self.get_isolated_runtime_cache(trace_df, isolated_trace_df)
        concurrent_df = trace_df[trace_df['num_concurrent_queries'] > 0]

        global_y = []
        global_x = []
        global_ir = []
        for i, rows in concurrent_df.groupby("query_idx"):
            if i not in self.isolated_rt_cache:
                continue
            isolated_rt = self.isolated_rt_cache[i]
            concurrent_rt = rows["runtime"].values
            if use_train:
                num_concurrency = rows["num_concurrent_queries_train"].values
            else:
                num_concurrency = rows["num_concurrent_queries"].values
            if len(num_concurrency) < 10:
                continue
            global_y.append(concurrent_rt)
            global_x.append(num_concurrency)
            global_ir.append(np.ones(len(num_concurrency)) * isolated_rt)
            fit, _ = optimization.curve_fit(queueing_func,
                                            (num_concurrency, np.ones(len(num_concurrency)) * isolated_rt),
                                            concurrent_rt,
                                            np.array([5, 0.1, 20]))
            self.a1[i] = fit[0]
            self.a2[i] = fit[1]
            self.b1[i] = fit[2]
        global_y = np.concatenate(global_y)
        global_x = np.concatenate(global_x)
        global_ir = np.concatenate(global_ir)
        fit, _ = optimization.curve_fit(queueing_func,
                                        (global_x, global_ir),
                                        global_y,
                                        np.array([5, 0.1, 20]))
        self.a1_global = fit[0]
        self.a2_global = fit[1]
        self.b1_global = fit[2]

    def predict(self, eval_trace_df, use_global=False, use_train=True):
        predictions = dict()
        labels = dict()
        for i, rows in eval_trace_df.groupby("query_idx"):
            if i not in self.isolated_rt_cache or i not in self.a1:
                continue
            isolated_rt = self.isolated_rt_cache[i]
            label = rows["runtime"].values
            labels[i] = label
            if use_train:
                num_concurrency = rows["num_concurrent_queries_train"].values
            else:
                num_concurrency = rows["num_concurrent_queries"].values
            x = (num_concurrency, np.ones(len(num_concurrency)) * isolated_rt)
            if use_global:
                pred = queueing_func(x, self.a1_global, self.a2_global, self.b1_global)
            else:
                pred = queueing_func(x, self.a1[i], self.a2[i], self.b1[i])
            pred = np.maximum(pred, 0.001)
            predictions[i] = pred
        return predictions, labels


class SimpleLinearReg(ConcurPredictor):
    """
    Simple linear regression model for runtime prediction with concurrency
    simple linear regression considers runtime = (b + k * num_concurrency) * isolated_runtime
    Use linear regression to find b, k
    """
    def __init__(self):
        super().__init__()
        self.isolated_rt_cache = dict()
        self.intercept_global = 0
        self.intercept = dict()
        self.slope_global = 0
        self.slope = dict()

    def train(self, trace_df, use_train=True, isolated_trace_df=None):
        self.get_isolated_runtime_cache(trace_df, isolated_trace_df)
        concurrent_df = trace_df[trace_df['num_concurrent_queries'] > 0]

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
            global_y.append(concurrent_rt/isolated_rt)
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

    def predict(self, eval_trace_df, use_global=False, use_train=True):
        predictions = dict()
        labels = dict()
        for i, rows in eval_trace_df.groupby("query_idx"):
            if i not in self.isolated_rt_cache:
                continue
            isolated_rt = self.isolated_rt_cache[i]
            label = rows["runtime"].values
            labels[i] = label
            if use_train:
                num_concurrency = rows["num_concurrent_queries_train"].values
            else:
                num_concurrency = rows["num_concurrent_queries"].values
            if use_global:
                pred = (num_concurrency * self.slope_global + self.intercept_global) * isolated_rt
            else:
                pred = (num_concurrency * self.slope[i] + self.intercept[i]) * isolated_rt
            pred = np.maximum(pred, 0.001)
            predictions[i] = pred
        return predictions, labels
