import copy
from typing import List, Any, Dict, Optional, Tuple
import json
import os
import numpy.typing as npt
import numpy as np
import pandas as pd


def _fill_binds(sql: str, binds: List[Any]):
    result = copy.deepcopy(sql)
    for i in range(len(binds) - 1, -1, -1):
        if binds[i] is None:
            return result
        if type(binds[i]) == int or type(binds[i]) == float:
            result = result.replace(f":{i + 1}", str(binds[i]))
        elif type(binds[i]) == str:
            result = result.replace(f":{i + 1}", binds[i])
    return result


def get_all_query_template(query_template_path: str) -> Dict[int, str]:
    query_template: Dict[int, str] = dict()
    for file in os.listdir(query_template_path):
        if file.endswith(".sql"):
            query_no = int(file.split(".sql")[0])
            with open(os.path.join(query_template_path, file), "r") as f:
                query = f.read().strip()
            query_template[query_no] = query
    return query_template


def convert_to_trace_df(cab_trace_path: str,
                        query_template_path: str,
                        save_dir: Optional[str] = None) -> Optional[Tuple[int, List[str], pd.DataFrame]]:
    with open(cab_trace_path, "r") as f:
        cab_trace = json.load(f)
    scale_factor: int = cab_trace['scale_factor']
    query_template = get_all_query_template(query_template_path)

    all_unique_queries: List[str] = []
    unique_query_templates: List[int] = []
    query_start_time: List[float] = []
    g_offset_since_start_s: List[float] = []
    all_queries: List[str] = []
    all_query_idx: List[int] = []
    all_query_template_idx: List[int] = []
    first_start_time = cab_trace['queries'][0]['start'] / 1000
    for query in cab_trace['queries']:
        q_id = query['query_id']
        if q_id not in query_template:
            continue
        start_s = query['start'] / 1000
        query_start_time.append(start_s)
        g_offset_since_start_s.append(start_s - first_start_time)
        query_str = _fill_binds(query_template[q_id], query['arguments'])
        all_query_template_idx.append(q_id)
        all_queries.append(query_str)
        if query_str not in all_unique_queries:
            all_unique_queries.append(query_str)
            query_idx = len(all_unique_queries) - 1
            unique_query_templates.append(q_id)
        else:
            query_idx = all_unique_queries.index(query_str)
        all_query_idx.append(query_idx)

    trace_df = pd.DataFrame(
        {
            "query_idx": all_query_idx,
            "run_time_s": [100.0] * len(all_query_idx),
            "g_offset_since_start_s": g_offset_since_start_s,
            "start_s": query_start_time,
            "query_sql": all_queries,
            "query_template_idx": all_query_template_idx
        }
    )

    if save_dir:
        all_unique_query_path = os.path.join(save_dir, f"tpc_sf{scale_factor}_all_unique_queries.sql")
        with open(all_unique_query_path, "w") as f:
            for query in all_unique_queries:
                f.write(query)
                f.write('\n')
        unique_query_templates_path = os.path.join(save_dir, f"tpc_sf{scale_factor}_unique_query_templates.npy")
        np.save(unique_query_templates_path, np.asarray(unique_query_templates, dtype=int))
        trace_path = os.path.join(save_dir, f"tpc_sf{scale_factor}_query_trace.csv")
        trace_df.to_csv(trace_path, index=False)
    else:
        return scale_factor, all_unique_queries, trace_df


def get_num_queries_per_time_interval(
    rows: pd.DataFrame,
    time_gap: int = 10,
) -> (List, npt.NDArray, npt.NDArray):
    """
    Get the number of queries per time interval every {time_gap} minutes
    time_gap: provide aggregated stats every {time_interval} minutes, for current implementation
            please make it a number divisible by 60
    """
    rows = rows.sort_values("time_since_execution_s", ascending=True)
    time_intervals = []
    all_time_intervals = []
    for h in range(24):
        hour = str(h) if h >= 10 else f"0{h}"
        for m in range(0, 60, time_gap):
            minute = str(m) if m >= 10 else f"0{m}"
            time_intervals.append(f"{hour}:{minute}:00")
            all_time_intervals.append(h * 3600 + m * 60)

    start_time = rows["time_since_execution_s"].values
    all_queries_runtime = rows["run_time_s"].values
    end_time = start_time + all_queries_runtime

    num_queries = []
    num_concurrent_queries = []

    start_idx = 0  # the start index of the loop
    for t in all_time_intervals[1:]:
        end_idx = np.searchsorted(start_time, t)
        if end_idx == start_idx:
            num_queries.append(0)
            num_concurrent_queries.append(0)
            continue
        num_queries.append(end_idx - start_idx)
        num_concurrent_queries_cnt = 0
        for i in range(start_idx, end_idx):
            s = start_time[i]
            e = end_time[i]
            if i < len(start_time) - 1:
                ne = np.searchsorted(
                    start_time[i + 1:], e
                )  # number of queries start after s and before e
            else:
                ne = 0
            ns = np.sum(
                end_time[:i] > s
            )  # number of queries start before s and ends after s
            num_concurrent_queries_cnt += ne + ns
        num_concurrent_queries.append(
            num_concurrent_queries_cnt / (end_idx - start_idx)
        )
        start_idx = end_idx

    return (
        time_intervals,
        num_queries,
        num_concurrent_queries,
        all_queries_runtime,
    )


