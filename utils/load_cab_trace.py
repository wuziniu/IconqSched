import copy
from typing import List, Any, Dict, Optional, Tuple
import json
import os
import pandas as pd


def _fill_binds(sql: str, binds: List[Any]):
    result = copy.deepcopy(sql)
    for i in range(len(binds)):
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
    query_start_time: List[float] = []
    g_offset_since_start_s: List[float] = []
    all_queries: List[str] = []
    all_query_idx: List[int] = []
    all_query_template_idx: List[int] = []
    first_start_time = cab_trace['queries'][0]['start'] / 1000
    for query in cab_trace['queries']:
        q_id = query['query_id']
        if q_id not in query_template:
            # print(f"{q_id} does not exist in query templates")
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
        else:
            query_idx = all_unique_queries.index(query_str)
        all_query_idx.append(query_idx)

    trace_df = pd.DataFrame(
        {
            "query_idx": all_query_idx,
            "run_time_s": [1.0] * len(all_query_idx),
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
        trace_path = os.path.join(save_dir, f"tpc_sf{scale_factor}_query_trace.csv")
        trace_df.to_csv(trace_path, index=False)
    else:
        return scale_factor, all_unique_queries, trace_df

