import numpy as np


class ConcurPredictor:
    def __init__(self):
        self.isolated_rt_cache = dict()
        self.average_rt_cache = dict()

    def get_isolated_runtime_cache(
        self, trace_df, isolated_trace_df=None, get_avg_runtime=False
    ):
        if isolated_trace_df is None:
            isolated_trace_df = trace_df[trace_df["num_concurrent_queries"] == 0]
        # ignore queries that do not run insolation
        for i, rows in isolated_trace_df.groupby("query_idx"):
            self.isolated_rt_cache[i] = np.median(rows["runtime"])
        if get_avg_runtime:
            for i, rows in trace_df.groupby("query_idx"):
                self.average_rt_cache[i] = np.median(rows["runtime"])

    def train(self, trace_df):
        raise NotImplemented

    def predict(self, eval_trace_df, use_global):
        raise NotImplemented

    def evaluate_performance(self, eval_trace_df, use_global=False):
        predictions, labels = self.predict(eval_trace_df, use_global)
        pred_all = []
        labels_all = []
        result_overall = []
        result_per_query = dict()
        for i in predictions:
            result_per_query[i] = []
            pred_all.append(predictions[i])
            labels_all.append(labels[i])
            abs_error = np.abs(predictions[i] - labels[i])
            q_error = np.maximum(predictions[i] / labels[i], labels[i] / predictions[i])
            for p in [50, 90, 95]:
                result_per_query[i].append(np.percentile(abs_error, p))
                result_per_query[i].append(np.percentile(q_error, p))

        pred_all = np.concatenate(pred_all)
        labels_all = np.concatenate(labels_all)
        abs_error = np.abs(pred_all - labels_all)
        q_error = np.maximum(pred_all / labels_all, labels_all / pred_all)
        for p in [50, 90, 95]:
            p_a = np.percentile(abs_error, p)
            p_q = np.percentile(q_error, p)
            print(f"{p}% absolute error is {p_a}, q-error is {p_q}")
            result_overall.append(p_a)
            result_overall.append(p_q)
        return result_overall, result_per_query
