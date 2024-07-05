import copy

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import Optional, Union, List, Tuple


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
