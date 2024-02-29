import numpy as np
import scipy.optimize as optimization
from models.concurrency.base_model import ConcurPredictor
from parser.utils import load_json, dfs_cardinality, estimate_scan_in_mb


def simple_queueing_func(x, a1, a2, b1):
    """
    a1 represents the average exec-time of a random query under concurrency
    b1 represents the max level of concurrency in a system
    a2 represents the average impact on a query's runtime when executed concurrently with other queries
    """
    num_concurrency, isolated_runtime = x
    return (a1 * np.maximum(num_concurrency - b1, 0)) + (
        1 + a2 * np.minimum(num_concurrency, b1)
    ) * isolated_runtime


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
        self.use_train = True
        self.a1_global = 0
        self.a1 = dict()
        self.b1_global = 0
        self.b1 = dict()
        self.a2_global = 0
        self.a2 = dict()

    def train(self, trace_df, use_train=True, isolated_trace_df=None):
        self.use_train = use_train
        self.get_isolated_runtime_cache(trace_df, isolated_trace_df)
        concurrent_df = trace_df[trace_df["num_concurrent_queries"] > 0]

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
            fit, _ = optimization.curve_fit(
                simple_queueing_func,
                (num_concurrency, np.ones(len(num_concurrency)) * isolated_rt),
                concurrent_rt,
                np.array([5, 0.1, 20]),
            )
            self.a1[i] = fit[0]
            self.a2[i] = fit[1]
            self.b1[i] = fit[2]
        global_y = np.concatenate(global_y)
        global_x = np.concatenate(global_x)
        global_ir = np.concatenate(global_ir)
        fit, _ = optimization.curve_fit(
            simple_queueing_func,
            (global_x, global_ir),
            global_y,
            np.array([5, 0.1, 20]),
        )
        self.a1_global = fit[0]
        self.a2_global = fit[1]
        self.b1_global = fit[2]

    def predict(self, eval_trace_df, use_global=False):
        predictions = dict()
        labels = dict()
        for i, rows in eval_trace_df.groupby("query_idx"):
            if i not in self.isolated_rt_cache or i not in self.a1:
                continue
            isolated_rt = self.isolated_rt_cache[i]
            label = rows["runtime"].values
            labels[i] = label
            if self.use_train:
                num_concurrency = rows["num_concurrent_queries_train"].values
            else:
                num_concurrency = rows["num_concurrent_queries"].values
            x = (num_concurrency, np.ones(len(num_concurrency)) * isolated_rt)
            if use_global:
                pred = simple_queueing_func(
                    x, self.a1_global, self.a2_global, self.b1_global
                )
            else:
                pred = simple_queueing_func(x, self.a1[i], self.a2[i], self.b1[i])
            pred = np.maximum(pred, 0.001)
            predictions[i] = pred
        return predictions, labels


def interaction_func(
    x, q1, i1, i2, c1, m1, m2, m3, cm1, r1, r2, max_concurrency, avg_io_speed, memory_size
):
    """
    An analytical function that can consider 3 types of resource sharing/contention: IO, memory, CPU
    x:: input tuple containing:
        isolated_runtime: the isolated runtime without concurrency of a query
        avg_runtime: average or median observed runtime of a query under any concurrency
        num_concurrency: number of concurrent queries running with this query
        sum_concurrent_runtime: sum of the estimated runtime of all queries concurrently running with this query (CPU)
        est_scan: estimated MB of data that this query will need to scan (IO)
        est_concurrent_scan: estimated MB of data that the concurrently running queries will need to scan (IO)
        scan_sharing_percentage: estimated percentage of data in cache (sharing) according to concurrent queries
        max_est_card: maximum estimated cardinality in the query plan of this query (reflect peak memory usage)
        avg_est_card: average estimated cardinality in the query plan of this query (reflect average memory usage)
        max_concurrent_card: maximum estimated cardinality for all concurrent queries
        avg_concurrent_card: average estimated cardinality for all concurrent queries
    TODO: adding memory and CPU information
    """
    (
        isolated_runtime,
        avg_runtime,
        num_concurrency,
        sum_concurrent_runtime,
        est_scan,
        est_concurrent_scan,
        scan_sharing_percentage,
        max_est_card,
        avg_est_card,
        max_concurrent_card,
        avg_concurrent_card,
    ) = x
    # fraction of running queries (as opposed to queueing queries)
    running_frac = np.minimum(num_concurrency, max_concurrency) / np.maximum(
        num_concurrency, 1
    )
    # estimate queueing time of a query based on the sum of concurrent queries' run time
    queueing_time = (
        q1
        * (
            np.maximum(num_concurrency - max_concurrency, 0)
            / np.maximum(num_concurrency, 1)
        )
        * sum_concurrent_runtime
    )
    # estimate io_speed of a query assuming each query has a base io_speed of i1 + the io speed due to contention
    io_speed = i1 + avg_io_speed / np.minimum(
        np.maximum(num_concurrency, 1), max_concurrency
    )
    # estimate time speed on IO as the (estimated scan - data in cache) / estimated io_speed
    # use i2 to adjust the estimation error in est_scan and scan_sharing_percentage
    io_time = i2 * est_scan * (1 - scan_sharing_percentage) / io_speed
    # estimate the amount of CPU work/time as the weighted average of isolated_runtime and avg_runtime - io_time
    cpu_time_isolated = (r1 * isolated_runtime + r2 * avg_runtime) - io_time
    # estimate the amount of CPU work imposed by the concurrent queries (approximated by their estimate runtime)
    cpu_concurrent = (running_frac * sum_concurrent_runtime) / avg_runtime
    # estimate the amount of memory imposed by the concurrent queries
    max_mem_usage_perc = max_concurrent_card / (max_concurrent_card + max_est_card)
    avg_mem_usage_perc = avg_concurrent_card / (avg_concurrent_card + avg_est_card)
    memory_concurrent = np.log(
        m1 * np.maximum(max_concurrent_card + max_est_card - memory_size, 0.01) * max_mem_usage_perc
        + m2 * np.maximum(avg_concurrent_card + avg_est_card - memory_size, 0.01) * avg_mem_usage_perc
        + 0.0001
    ) * np.log(m1 * max_est_card + m2 * avg_est_card + 0.0001)
    memory_concurrent = np.maximum(memory_concurrent, 0)
    # estimate the CPU time of a query by considering the contention of CPU and memory of other queries
    cpu_time = (
        1
        + c1 * cpu_concurrent
        + m3 * memory_concurrent
        + cm1 * np.sqrt(cpu_concurrent * memory_concurrent)
    ) * cpu_time_isolated
    # final runtime of a query is estimated to be the queueing time + io_time + cpu_time
    return np.maximum(queueing_time + io_time + cpu_time, 0.01)


class ComplexFitCurve(ConcurPredictor):
    """
    Simple fit curve model for runtime prediction with concurrency
    runtime = queue_time(num_concurrency) + alpha(num_concurrency) * isolated_runtime
            = (a1 * max(num_concurrency-b1, 0)) + (1 + a2*min(num_concurrency, b1)) * isolated_runtime
    optimize a1, b1, b2
    """

    def __init__(self):
        super().__init__()
        self.isolated_rt_cache = dict()
        self.average_rt_cache = dict()
        self.query_info = dict()
        self.db_stats = None
        self.table_sizes = dict()
        self.table_sizes_by_index = dict()
        self.table_nrows_by_index = dict()
        self.table_column_map = dict()
        self.use_train = True
        self.is_column_store = False
        self.q1 = 0.5
        self.i1 = 10
        self.i2 = 2
        self.c1 = 0.2
        self.m1 = 0.5
        self.m2 = 0.5
        self.m3 = 0.1
        self.cm1 = 0.2
        self.r1 = 0.8
        self.r2 = 0.2
        self.max_concurrency = 10
        self.avg_io_speed = 200
        self.memory_size = 16000
        self.bound = optimization.Bounds(
            [0.1, 10, 0.01, 0.001, 0.001, 0.001, 0.001, 0.001, 0.5, 0.05, 2, 20, 10000],
            [1, 200, 2, 1, 0.9, 0.9, 0.5, 0.5, 0.95, 0.4, 20, 2000, 50000],
        )

    def _compute_table_size(self):
        for col in self.db_stats["column_stats"]:
            table = col["tablename"]
            if table not in self.table_column_map:
                self.table_sizes[table] = 0
                self.table_column_map[table] = []
            self.table_column_map[table].append(col["attname"])
            if col["avg_width"] is not None and col["avg_width"] > 0:
                self.table_sizes[table] += col["avg_width"]
        all_table_names = [t["relname"] for t in self.db_stats["table_stats"]]
        for table in self.table_sizes:
            if table in all_table_names:
                idx = all_table_names.index(table)
                num_tuples = self.db_stats["table_stats"][idx]["reltuples"]
                self.table_nrows_by_index[idx] = num_tuples
                size_in_mb = (num_tuples * self.table_sizes[table]) / (1024 * 1024)
                self.table_sizes[table] = size_in_mb
                self.table_sizes_by_index[idx] = size_in_mb

    def pre_process_queries(
        self, parsed_queries_path, with_width=True, use_true_card=False
    ):
        plans = load_json(parsed_queries_path, namespace=False)
        self.db_stats = plans["database_stats"]
        self._compute_table_size()
        self.query_info = dict()
        for i in range(len(plans["sql_queries"])):
            curr_query_info = dict()
            curr_query_info["sql"] = plans["sql_queries"][i]
            all_cardinality = []
            dfs_cardinality(
                plans["parsed_plans"][i], all_cardinality, with_width, use_true_card
            )
            curr_query_info["all_cardinality"] = all_cardinality
            est_scan, est_scan_per_table = estimate_scan_in_mb(
                self.db_stats,
                plans["parsed_queries"][i],
                use_true_card,
                self.is_column_store,
            )
            curr_query_info["est_scan"] = est_scan
            curr_query_info["est_scan_per_table"] = est_scan_per_table
            self.query_info[i] = curr_query_info

    def estimate_data_share_percentage(self, idx, concur_info, pre_exec_info=None):
        # TODO: make it smarter by considering buffer pool behavior
        curr_scan = self.query_info[idx]["est_scan_per_table"]
        curr_total_scan = self.query_info[idx]["est_scan"]
        if pre_exec_info is not None:
            concur_info = concur_info + pre_exec_info
        all_shared_scan = 0
        for table in curr_scan:
            table_size = self.table_sizes_by_index[table]
            table_shared_scan = 0
            for c in concur_info:
                concur_scan = self.query_info[c[0]]["est_scan_per_table"]
                if table in concur_scan:
                    concur_scan_perc = concur_scan[table] / table_size
                    overlap_scan = concur_scan_perc * curr_scan[table]
                    table_shared_scan += overlap_scan
            table_shared_scan = min(table_shared_scan, curr_scan[table])
            all_shared_scan += table_shared_scan
        return min(all_shared_scan / curr_total_scan, 1.0)

    def featurize_data(self, concurrent_df):
        global_y = []
        global_isolated_runtime = []
        global_avg_runtime = []
        global_num_concurrency = []
        global_sum_concurrent_runtime = []
        global_est_scan = []
        global_est_concurrent_scan = []
        global_scan_sharing_percentage = []
        global_max_est_card = []
        global_avg_est_card = []
        global_max_concurrent_card = []
        global_avg_concurrent_card = []
        global_query_idx = dict()
        start = 0
        for i, rows in concurrent_df.groupby("query_idx"):
            if (
                i not in self.isolated_rt_cache
                or i not in self.query_info
                or i not in self.average_rt_cache
            ):
                continue
            concurrent_rt = rows["runtime"].values
            query_info = self.query_info[i]
            n_rows = len(rows)
            if self.use_train:
                num_concurrency = rows["num_concurrent_queries_train"].values
                concur_info = rows["concur_info_train"].values
            else:
                num_concurrency = rows["num_concurrent_queries"].values
                concur_info = rows["concur_info"].values
            pre_exec_info = rows["pre_exec_info"].values

            global_query_idx[i] = (start, start + n_rows)
            start += n_rows
            global_y.append(concurrent_rt)
            global_isolated_runtime.append(np.ones(n_rows) * self.isolated_rt_cache[i])
            global_avg_runtime.append(np.ones(n_rows) * self.average_rt_cache[i])
            global_num_concurrency.append(num_concurrency)
            global_est_scan.append(np.ones(n_rows) * query_info["est_scan"])
            global_max_est_card.append(
                np.ones(n_rows) * np.max(query_info["all_cardinality"]) / (1024 * 1024)
            )
            global_avg_est_card.append(
                np.ones(n_rows) * np.average(query_info["all_cardinality"]) / (1024 * 1024)
            )
            for j in range(n_rows):
                sum_concurrent_runtime = 0
                sum_concurrent_scan = 0
                concurrent_card = []
                for c in concur_info[j]:
                    if c[0] in self.average_rt_cache:
                        sum_concurrent_runtime += self.average_rt_cache[c[0]]
                    else:
                        print(c[0])
                    if c[0] in self.query_info:
                        sum_concurrent_scan += self.query_info[c[0]]["est_scan"]
                        concurrent_card.extend(self.query_info[c[0]]["all_cardinality"])
                    else:
                        print(c[0])

                global_sum_concurrent_runtime.append(sum_concurrent_runtime)
                global_est_concurrent_scan.append(sum_concurrent_scan)
                if len(concurrent_card) == 0:
                    global_max_concurrent_card.append(0)
                    global_avg_concurrent_card.append(0)
                else:
                    global_max_concurrent_card.append(np.max(concurrent_card) / (1024 * 1024))
                    global_avg_concurrent_card.append(
                        np.average(concurrent_card) / (1024 * 1024)
                    )
                global_scan_sharing_percentage.append(
                    self.estimate_data_share_percentage(
                        i, concur_info[j], pre_exec_info[j]
                    )
                )

        global_y = np.concatenate(global_y)
        global_isolated_runtime = np.concatenate(global_isolated_runtime)
        global_avg_runtime = np.concatenate(global_avg_runtime)
        global_num_concurrency = np.concatenate(global_num_concurrency)
        global_est_scan = np.concatenate(global_est_scan)
        global_max_est_card = np.concatenate(global_max_est_card)
        global_avg_est_card = np.concatenate(global_avg_est_card)
        global_sum_concurrent_runtime = np.asarray(global_sum_concurrent_runtime)
        global_est_concurrent_scan = np.asarray(global_est_concurrent_scan)
        global_max_concurrent_card = np.asarray(global_max_concurrent_card)
        global_avg_concurrent_card = np.asarray(global_avg_concurrent_card)
        global_scan_sharing_percentage = np.asarray(global_scan_sharing_percentage)
        feature = (
            global_isolated_runtime,
            global_avg_runtime,
            global_num_concurrency,
            global_sum_concurrent_runtime,
            global_est_scan,
            global_est_concurrent_scan,
            global_scan_sharing_percentage,
            global_max_est_card,
            global_avg_est_card,
            global_max_concurrent_card,
            global_avg_concurrent_card,
        )
        return feature, global_y, global_query_idx

    def train(self, trace_df, use_train=True, isolated_trace_df=None):
        self.use_train = use_train
        self.get_isolated_runtime_cache(
            trace_df, isolated_trace_df, get_avg_runtime=True
        )
        concurrent_df = trace_df[trace_df["num_concurrent_queries"] > 0]
        feature, label, _ = self.featurize_data(concurrent_df)

        initial_param_value = np.array(
            [
                self.q1,
                self.i1,
                self.i2,
                self.c1,
                self.m1,
                self.m2,
                self.m3,
                self.cm1,
                self.r1,
                self.r2,
                self.max_concurrency,
                self.avg_io_speed,
                self.memory_size
            ]
        )
        fit, _ = optimization.curve_fit(
            interaction_func,
            feature,
            label,
            initial_param_value,
            bounds=self.bound,
            jac="3-point",
            method="trf",
            loss="soft_l1",
        )
        self.q1 = fit[0]
        self.i1 = fit[1]
        self.i2 = fit[2]
        self.c1 = fit[3]
        self.m1 = fit[4]
        self.m2 = fit[5]
        self.m3 = fit[6]
        self.cm1 = fit[7]
        self.r1 = fit[8]
        self.r2 = fit[9]
        self.max_concurrency = fit[10]
        self.avg_io_speed = fit[11]
        self.memory_size = fit[12]

    def predict(self, eval_trace_df, use_global=False, return_per_query=True):
        feature, labels, query_idx = self.featurize_data(eval_trace_df)
        preds = interaction_func(
            feature,
            self.q1,
            self.i1,
            self.i2,
            self.c1,
            self.m1,
            self.m2,
            self.m3,
            self.cm1,
            self.r1,
            self.r2,
            self.max_concurrency,
            self.avg_io_speed,
            self.memory_size
        )
        if return_per_query:
            preds_per_query = dict()
            labels_per_query = dict()
            for i in query_idx:
                start, end = query_idx[i]
                preds_per_query[i] = preds[start:end]
                labels_per_query[i] = labels[start:end]
            return preds_per_query, labels_per_query
        else:
            return preds, labels
