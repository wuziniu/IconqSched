import copy

import numpy as np
import pandas as pd
from typing import Optional, Union, List, Tuple, Dict


def report_performance(
    rt: np.ndarray, idx: Optional[Union[np.ndarray, List[int]]] = None
) -> None:
    if idx is None:
        runtime = rt
    else:
        runtime = rt[idx]
    print(
        np.mean(runtime),
        np.percentile(runtime, 50),
        np.percentile(runtime, 90),
        np.percentile(runtime, 95),
        np.percentile(runtime, 99),
    )


def report_performance_by_tpc_template(preds: Dict[int, List[float]],
                                       labels: Dict[int, List[float]],
                                       template_idx: np.ndarray
                                       ) -> Tuple[Dict[int, List[float]], Dict[int, List[float]]]:
    new_preds: Dict[int, List[float]] = dict()
    new_labels: Dict[int, List[float]] = dict()

    query_idx_by_template: Dict[int, List[int]] = dict()
    for q_id, t_id in enumerate(template_idx):
        if t_id not in query_idx_by_template:
            query_idx_by_template[t_id] = [q_id]
        else:
            query_idx_by_template[t_id].append(q_id)

    for t_id in sorted(list(query_idx_by_template.keys())):
        new_preds[t_id] = []
        new_labels[t_id] = []
        for q_id in query_idx_by_template[t_id]:
            if q_id in preds:
                new_preds[t_id].extend(preds[q_id])
                new_labels[t_id].extend(labels[q_id])
        if len(new_preds[t_id]) != 0:
            curr_preds = np.asarray(new_preds[t_id])
            curr_labels = np.asarray(new_labels[t_id])
            q_error = np.maximum(curr_preds/curr_labels, curr_labels/curr_preds)
            p50 = np.percentile(q_error, 50)
            p90 = np.percentile(q_error, 90)
            p99 = np.percentile(q_error, 95)
            print(f"For tpc-h template Q_{t_id}: p50 {p50}, p90 {p90}, p99 {p99}")
    return new_preds, new_labels


def load_and_report_stats(
    path: str, baseline: bool = False
) -> Tuple[np.ndarray, np.ndarray]:
    if path.endswith(".csv"):
        results = pd.read_csv(path)
        rt = results["exec_time"].values
        e2e = results["run_time_s"].values
    else:
        if not path.endswith("_"):
            path += "_"
        if baseline:
            rt = np.load(path + "sys_exec_time_baseline.npy")
            e2e = np.load(path + "e2e_runtime_baseline.npy")
        else:
            rt = np.load(path + "sys_exec_time_ours.npy")
            e2e = np.load(path + "e2e_runtime_ours.npy")
    idx = np.where(e2e < 0)[0]  # negative value means unfinished execution
    print("Execution time: ")
    report_performance(rt, idx)
    print("End-to-end time: ")
    report_performance(e2e, idx)
    return rt, e2e


def create_concurrent_df_from_results(
    trace_path: str,
    exec_time: np.ndarray,
    e2e_time: np.ndarray,
    save_path: Optional[str] = None,
    start_idx: int = 0,
) -> pd.DataFrame:
    # This function turns the results in the save format as training data
    trace = pd.read_csv(trace_path)
    trace = trace.iloc[start_idx:]
    queueing_time = np.maximum(e2e_time - exec_time, 0)
    # assert np.sum(queueing_time < 0) == 0, "some queries with negative queueing time"
    new_start_time = []
    start_s = trace["g_offset_since_start_s"].values
    start_s = start_s - np.min(start_s[start_s >= 0])
    query_idx = trace["query_idx"].values
    new_query_idx = []
    run_time_s = []
    e2e_time_s = []
    for i in range(len(exec_time)):
        if exec_time[i] < 0 or e2e_time[i] < 0:
            continue
        new_start_time.append(start_s[i] + queueing_time[i])
        run_time_s.append(exec_time[i])
        e2e_time_s.append(e2e_time[i])
        new_query_idx.append(query_idx[i])
    df = pd.DataFrame(
        {
            "query_idx": np.asarray(new_query_idx),
            "run_time_s": np.asarray(run_time_s),
            "e2e_time_s": np.asarray(e2e_time_s),
            "g_offset_since_start_s": np.asarray(new_start_time),
        }
    )
    df = df.sort_values(by=["g_offset_since_start_s"], ascending=True)
    if save_path:
        df.to_csv(save_path, index=False)
    return df


def realign_execution_start_time(path: str, inplace: bool = True) -> pd.DataFrame:
    results = pd.read_csv(path)
    g_offset_since_start_s = results["g_offset_since_start_s"].values
    g_offset_since_start_s = g_offset_since_start_s - np.min(
        g_offset_since_start_s[g_offset_since_start_s >= 0]
    )
    queueing_time = np.maximum(
        results["run_time_s"].values - results["exec_time"].values, 0
    )
    # assert np.sum(queueing_time < 0) == 0, "some queries with negative queueing time"
    results["g_offset_since_start_s"] = g_offset_since_start_s + queueing_time
    results["e2e_time_s"] = copy.deepcopy(results["run_time_s"].values)
    results["run_time_s"] = results["exec_time"].values
    results = results[results["exec_time"] > 0]
    results = results[results["error"] == False]
    results = results.sort_values(by=["g_offset_since_start_s"], ascending=True)
    if inplace:
        results.to_csv(path, index=False)
    return results
