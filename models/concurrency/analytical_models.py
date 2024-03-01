import numpy as np
from xgboost import XGBRegressor
import torch
import torch.optim as optim
from torch.nn.functional import l1_loss
import scipy.optimize as optimization
from torch.utils.data import DataLoader
from models.concurrency.base_model import ConcurPredictor
from models.concurrency.utils import QueryFeatureDataset, SimpleNet
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


def interaction_func_scipy(
    x,
    q1,
    i1,
    i2,
    c1,
    m1,
    m2,
    m3,
    cm1,
    r1,
    r2,
    max_concurrency,
    avg_io_speed,
    memory_size,
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
    # estimate the amount of memory load imposed by the concurrent queries
    max_mem_usage_perc = max_concurrent_card / (max_concurrent_card + max_est_card)
    avg_mem_usage_perc = avg_concurrent_card / (avg_concurrent_card + avg_est_card)
    memory_concurrent = np.log(
        m1
        * np.maximum(max_concurrent_card + max_est_card - memory_size, 0.01)
        * max_mem_usage_perc
        + m2
        * np.maximum(avg_concurrent_card + avg_est_card - memory_size, 0.01)
        * avg_mem_usage_perc
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


def interaction_func_torch(
    x,
    q1,
    i1,
    i2,
    c1,
    m1,
    m2,
    m3,
    cm1,
    r1,
    r2,
    max_concurrency,
    avg_io_speed,
    memory_size,
):
    # See interaction_func_scipy for explanation
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
    num_query = len(num_concurrency)
    running_frac = torch.minimum(num_concurrency, max_concurrency) / torch.maximum(
        num_concurrency, torch.tensor(1)
    )
    # estimate queueing time of a query based on the sum of concurrent queries' run time
    queueing_time = (
        q1
        * (
            torch.maximum(num_concurrency - max_concurrency, torch.tensor(0))
            / torch.maximum(num_concurrency, torch.tensor(1))
        )
        * sum_concurrent_runtime
    )
    # estimate io_speed of a query assuming each query has a base io_speed of i1 + the io speed due to contention
    io_speed = i1 + avg_io_speed / torch.minimum(
        torch.maximum(num_concurrency, torch.tensor(1)), max_concurrency
    )
    # estimate time speed on IO as the (estimated scan - data in cache) / estimated io_speed
    # use i2 to adjust the estimation error in est_scan and scan_sharing_percentage
    io_time = i2 * est_scan * (1 - scan_sharing_percentage) / io_speed
    # estimate the amount of CPU work/time as the weighted average of isolated_runtime and avg_runtime - io_time
    cpu_time_isolated = (r1 * isolated_runtime + r2 * avg_runtime) - io_time
    # estimate the amount of CPU work imposed by the concurrent queries (approximated by their estimate runtime)
    cpu_concurrent = (running_frac * sum_concurrent_runtime) / avg_runtime
    # estimate the amount of memory load imposed by the concurrent queries
    max_mem_usage_perc = max_concurrent_card / (max_concurrent_card + max_est_card)
    avg_mem_usage_perc = avg_concurrent_card / (avg_concurrent_card + avg_est_card)
    memory_concurrent = torch.log(
        m1
        * torch.maximum(max_concurrent_card + max_est_card - memory_size, torch.tensor(0) + 0.01)
        * max_mem_usage_perc
        + m2
        * torch.maximum(avg_concurrent_card + avg_est_card - memory_size, torch.tensor(0) + 0.01)
        * avg_mem_usage_perc
        + 0.0001
    ) * torch.log(m1 * max_est_card + m2 * avg_est_card + 0.0001)
    memory_concurrent = torch.maximum(memory_concurrent, torch.tensor(0))
    # estimate the CPU time of a query by considering the contention of CPU and memory of other queries
    cpu_time = (
        1
        + c1 * cpu_concurrent
        + m3 * memory_concurrent
        + cm1 * torch.sqrt(cpu_concurrent * memory_concurrent)
    ) * cpu_time_isolated
    # final runtime of a query is estimated to be the queueing time + io_time + cpu_time
    return torch.maximum(queueing_time + io_time + cpu_time, torch.tensor(0) + 0.01)


def fit_curve_loss_torch(x, y, params, constrain, loss_func="soft_l1", penalties=None):
    pred = interaction_func_torch(x, *params)
    lb = constrain.lb
    ub = constrain.ub
    if loss_func == "mae":
        loss = torch.abs(pred - y)
    elif loss_func == "mse":
        loss = (pred - y) ** 2
    elif loss_func == "soft_l1":
        loss = torch.sqrt(1 + (pred - y) ** 2) - 1
    else:
        assert False, f"loss func {loss_func} not implemented"
    loss = torch.mean(loss)
    for i, p in enumerate(params):
        if penalties is not None:
            penalty = penalties[i]
        else:
            penalty = 1
        pen = torch.exp(penalty * (p - ub[i])) + torch.exp(-1 * penalty * (p - lb[i]))
        loss += pen
    return loss


class ComplexFitCurve(ConcurPredictor):
    """
    Complex fit curve model for runtime prediction with concurrency
    See interaction_func_scipy for detailed analytical functions
    """

    def __init__(self, is_column_store=False, opt_method='scipy'):
        """

        :param is_column_store:
        :param opt_method:
        """
        # indicate whether the DBMS is a column_store
        #
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
        self.is_column_store = is_column_store
        self.opt_method = opt_method
        self.batch_size = 1024
        self.analytic_params = [0.5, 20, 2, 0.2, 0.5, 0.5, 0.1, 0.2, 0.8, 0.2, 10, 200, 16000]
        self.bound = optimization.Bounds(
            [0.1, 10, 0.01, 0.001, 0.001, 0.001, 0.001, 0.001, 0.5, 0.05, 2, 20, 10000],
            [1, 200, 2, 1, 0.9, 0.9, 0.5, 0.5, 0.95, 0.4, 20, 2000, 50000],
        )
        self.constrain = optimization.Bounds(
            [0.1, 10, 0.1, 0.01, 0.01, 0.01, 0.01, 0.1, 0.5, 0.05, 2, 20, 10000],
            [1, 200, 2, 1, 0.9, 0.9, 0.5, 0.5, 0.95, 0.4, 20, 2000, 50000],
        )
        self.penalty = [100, 0.1, 100, 100, 100, 100, 100, 100, 100, 100, 1, 0.1, 0.01]
        self.loss_func = "soft_l1"
        self.model = None

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
                np.ones(n_rows)
                * np.average(query_info["all_cardinality"])
                / (1024 * 1024)
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
                    global_max_concurrent_card.append(
                        np.max(concurrent_card) / (1024 * 1024)
                    )
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
        if self.opt_method == "torch" or self.opt_method == "nn":
            feature = list(feature)
            for i in range(len(feature)):
                feature[i] = torch.from_numpy(feature[i])
            feature = tuple(feature)
            global_y = torch.from_numpy(global_y)
        return feature, global_y, global_query_idx

    def train(self, trace_df, use_train=True, isolated_trace_df=None):
        self.use_train = use_train
        self.get_isolated_runtime_cache(
            trace_df, isolated_trace_df, get_avg_runtime=True
        )
        concurrent_df = trace_df[trace_df["num_concurrent_queries"] > 0]
        feature, label, _ = self.featurize_data(concurrent_df)

        initial_param_value = np.asarray(self.analytic_params)
        if self.opt_method == "scipy":
            fit, _ = optimization.curve_fit(
                interaction_func_scipy,
                feature,
                label,
                initial_param_value,
                bounds=self.bound,
                jac="3-point",
                method="trf",
                loss="soft_l1",
                verbose=1
            )
            self.analytic_params = list(fit)
        elif self.opt_method == "torch":
            torch_analytic_params = []
            torch_analytic_params_lr = []
            for p in self.analytic_params:
                if p == 10:
                    t_p = torch.tensor(float(p), requires_grad=False)
                else:
                    t_p = torch.tensor(float(p), requires_grad=True)
                torch_analytic_params.append(t_p)
                torch_analytic_params_lr.append({'params': t_p, 'lr': 0.01 * p ** 0.3})
            optimizer = optim.Adam(torch_analytic_params_lr, weight_decay=2e-5)
            dataset = QueryFeatureDataset(feature, label)
            train_dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)
            for epoch in range(200):
                for X, y in train_dataloader:
                    optimizer.zero_grad()
                    loss = fit_curve_loss_torch(X, y, torch_analytic_params,
                                                self.constrain, loss_func=self.loss_func, penalties=self.penalty)
                    loss.backward()
                    optimizer.step()
                if epoch % 10 == 0:
                    print(epoch, loss.item())
                    print(torch_analytic_params)
            for i in range(len(self.analytic_params)):
                self.analytic_params[i] = torch_analytic_params[i].detach()
        elif self.opt_method == "nn":
            dataset = QueryFeatureDataset(feature, label)
            train_dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)
            self.model = SimpleNet(len(feature))
            optimizer = optim.Adam(self.model.parameters(), lr=0.01, weight_decay=2e-5)
            for epoch in range(200):
                for X, y in train_dataloader:
                    X = torch.stack(X).float()
                    X = torch.transpose(X, 0, 1)
                    optimizer.zero_grad()
                    pred = self.model(X)
                    pred = pred.reshape(-1)
                    loss = l1_loss(pred, y)
                    loss.backward()
                    optimizer.step()
                if epoch % 10 == 0:
                    print(epoch, loss.item())
        elif self.opt_method == "xgboost":
            feature = np.stack(feature).T
            model = XGBRegressor(
                n_estimators=1000,
                max_depth=8,
                eta=0.2,
                subsample=1.0,
                eval_metric="mae",
                early_stopping_rounds=100,
            )
            train_idx = np.random.choice(
                len(feature), size=int(0.8 * len(feature)), replace=False
            )
            val_idx = [i for i in range(len(feature)) if i not in train_idx]
            model.fit(
                feature[train_idx],
                label[train_idx],
                eval_set=[(feature[val_idx], label[val_idx])],
                verbose=False,
            )
            self.model = model
        else:
            assert False, f"unrecognized optimization method {self.opt_method}"

    def predict(self, eval_trace_df, use_global=False, return_per_query=True):
        feature, labels, query_idx = self.featurize_data(eval_trace_df)
        if self.opt_method == "scipy":
            preds = interaction_func_scipy(
                feature,
                *self.analytic_params
            )
        elif self.opt_method == "torch":
            preds = interaction_func_torch(
                feature,
                *self.analytic_params
            )
            preds = preds.numpy()
            labels = labels.numpy()
        elif self.opt_method == "nn":
            feature = torch.stack(feature).float()
            feature = torch.transpose(feature, 0, 1)
            preds = self.model(feature)
            preds = preds.reshape(-1)
            preds = preds.detach().numpy()
            labels = labels.numpy()
        elif self.opt_method == "xgboost":
            feature = np.stack(feature).T
            preds = self.model.predict(feature)
            preds = np.maximum(preds, 0.001)
        else:
            assert False, f"unrecognized optimization method {self.opt_method}"
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
