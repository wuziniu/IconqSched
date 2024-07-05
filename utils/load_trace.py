import pandas as pd
import os
import copy
from typing import List, Union, Optional, Tuple


def load_trace(
    directory: str,
    num_client: int = 10,
    concat: bool = True,
    version: Optional[int] = None,
) -> Union[
    Tuple[pd.DataFrame, pd.DataFrame], Tuple[List[pd.DataFrame], List[pd.DataFrame]]
]:
    # Load trace data.
    raw_trace = []
    trace = []
    starting_ts = None
    starting_s = None
    for i in range(num_client):
        if version is None:
            file = os.path.join(directory, f"trace_client_{i}.csv")
        else:
            file = os.path.join(directory, f"trace_client_{i}_{version}.csv")
        if not os.path.exists(file):
            if version is None:
                file = os.path.join(directory, f"repeating_olap_batch_{i}.csv")
            else:
                file = os.path.join(
                    directory, f"repeating_olap_batch_{i}_{version}.csv"
                )
        if not os.path.exists(file):
            continue
        df = pd.read_csv(file)
        df["timestamp"] = pd.to_datetime(df["timestamp"])

        raw_trace.append(df)
        this_min = df["timestamp"].min()
        if "time_since_execution_s" in df.columns:
            this_min_s = df["time_since_execution_s"].min()
            if starting_s is None:
                starting_s = this_min_s
            elif this_min_s < starting_s:
                starting_s = this_min_s
        if starting_ts is None:
            starting_ts = this_min
        elif this_min < starting_ts:
            starting_ts = this_min

        df2 = df.copy()
        df2["g_offset_since_start"] = df2["timestamp"] - starting_ts
        if "time_since_execution_s" in df2.columns:
            df2["g_offset_since_start_s"] = df2["time_since_execution_s"] - starting_s
        else:
            df2["g_offset_since_start_s"] = df2[
                "g_offset_since_start"
            ].dt.total_seconds()
        df2 = df2.sort_values(by=["g_offset_since_start_s"])
        initial_start = df2["g_offset_since_start_s"].iloc[0]
        df2["g_issue_gap_s"] = (
            df2["g_offset_since_start_s"]
            - df2["g_offset_since_start_s"].shift(periods=1)
        ).fillna(initial_start)
        trace.append(df2)

    if concat:
        raw_trace = pd.concat(raw_trace, ignore_index=True)
        raw_trace = raw_trace.sort_values("time_since_execution_s", ascending=True)
        trace = pd.concat(trace, ignore_index=True)
        trace = trace.sort_values("g_offset_since_start_s", ascending=True)
    return raw_trace, trace


def load_trace_all_version(
    directory: str, num_client: int = 10, concat: bool = True
) -> Union[
    Tuple[pd.DataFrame, pd.DataFrame], Tuple[List[pd.DataFrame], List[pd.DataFrame]]
]:
    exist_versions = []
    for v in range(2, 20):
        if os.path.exists(os.path.join(directory, f"trace_client_0_{v}.csv")):
            exist_versions.append(v)
        elif os.path.exists(os.path.join(directory, f"repeating_olap_batch_0_{v}.csv")):
            exist_versions.append(v)
    all_raw_trace = []
    all_trace = []
    raw_trace, trace = load_trace(directory, num_client, concat, version=None)
    all_raw_trace.append(raw_trace)
    all_trace.append(trace)
    for v in exist_versions:
        raw_trace, trace = load_trace(directory, num_client, concat, version=v)
        all_raw_trace.append(raw_trace)
        all_trace.append(trace)
    return all_raw_trace, all_trace


def load_all_trace_from_dir(directory: str) -> List[pd.DataFrame]:
    all_trace = []
    for file in os.listdir(directory):
        if file.endswith(".csv"):
            file = os.path.join(directory, file)
            df = pd.read_csv(file)
            df = df[df["run_time_s"] > 0]
            all_trace.append(df)
    return all_trace


def create_concurrency_dataset(
    trace: pd.DataFrame,
    engine: Optional[str] = None,
    pre_exec_interval: Optional[float] = None,
) -> pd.DataFrame:
    query_idx = []
    runtime = []
    start_time = []
    end_time = []
    concur_info = []
    concur_info_train = []
    pre_exec_info = []
    num_concurrent_queries = []
    num_concurrent_queries_train = []

    unfinished_queries = dict()
    pre_executed_queries = dict()
    for i in range(len(trace)):
        row = trace.iloc[i]
        if engine is not None and "engine" in trace.columns and row["engine"] != engine:
            continue
        query_idx.append(row["query_idx"])
        runtime.append(row["run_time_s"])
        curr_start_t = row["g_offset_since_start_s"]
        if "exec_time" in trace.columns:
            curr_end_t = curr_start_t + row["exec_time"]
        else:
            curr_end_t = curr_start_t + row["run_time_s"]
        start_time.append(curr_start_t)
        end_time.append(curr_end_t)

        cur_concur_info = []
        cur_pre_exec_info = []
        finished_key = []
        # adding unfinished queries (e.g., concurrently running) to the info
        for key in unfinished_queries:
            indx, start_t, end_t = unfinished_queries[key]
            if end_t <= curr_start_t:
                finished_key.append(key)
                if pre_exec_interval:
                    pre_executed_queries[key] = unfinished_queries[key]
            else:
                cur_concur_info.append((indx, start_t, end_t))
        for key in finished_key:
            unfinished_queries.pop(key)
        pre_exec_remove = []
        # adding just finished queries to the info
        if pre_exec_interval:
            for key in pre_executed_queries:
                indx, start_t, end_t = pre_executed_queries[key]
                if end_t <= curr_start_t - pre_exec_interval:
                    pre_exec_remove.append(key)
                else:
                    cur_pre_exec_info.append((indx, start_t, end_t))
        for key in pre_exec_remove:
            pre_executed_queries.pop(key)
        unfinished_queries[i] = (row["query_idx"], curr_start_t, curr_end_t)
        concur_info_train.append(copy.deepcopy(cur_concur_info))
        num_concurrent_queries_train.append(len(cur_concur_info))
        pre_exec_info.append(cur_pre_exec_info)
        for j in range(i + 1, i + 100):
            if j >= len(trace):
                break
            next_row = trace.iloc[j]
            next_start_t = next_row["g_offset_since_start_s"]
            if "exec_time" in trace.columns:
                next_end_t = next_start_t + next_row["exec_time"]
            else:
                next_end_t = next_start_t + next_row["run_time_s"]
            if next_start_t >= curr_end_t:
                break
            cur_concur_info.append((next_row["query_idx"], next_start_t, next_end_t))
        concur_info.append(cur_concur_info)
        num_concurrent_queries.append(len(cur_concur_info))

    concurrency_df = pd.DataFrame(
        {
            "query_idx": query_idx,
            "runtime": runtime,
            "start_time": start_time,
            "end_time": end_time,
            "pre_exec_info": pre_exec_info,
            "concur_info": concur_info,
            "num_concurrent_queries": num_concurrent_queries,
            "concur_info_train": concur_info_train,
            "num_concurrent_queries_train": num_concurrent_queries_train,
        }
    )
    for col in ["ground_truth_runtime", "exec_time", "run_time_s"]:
        if col in trace.columns:
            concurrency_df[col] = trace[col]
    concurrency_df = concurrency_df.reset_index()
    return concurrency_df
