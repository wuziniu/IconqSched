import os
import numpy as np
from models.concurrency.baselines.gcn_dbconnection import Database
from models.concurrency.baselines.gcn_nodeutils import (
    extract_plan,
    add_across_plan_relations,
)
from parser.utils import load_json
from utils.load_brad_trace import load_trace_all_version
from models.feature.single_xgboost_feature import find_top_k_operators


def generate_graph(
    concurrency_df,
    parsed_run,
    all_optype,
    knobs,
    run_id,
    time_interval=600,
    save_dir=None,
):
    parsed_plans = parsed_run["parsed_plans"]
    column_stats = parsed_run["database_stats"]["column_stats"]
    column_table_mapping = [col["tablename"] for col in column_stats]
    all_start_time = concurrency_df["time_since_execution_s"].values
    start_idx = 0
    curr_end_time = all_start_time[0] + time_interval
    wid = 0
    while curr_end_time <= all_start_time[-1]:
        end_idx = np.searchsorted(all_start_time, curr_end_time)
        if end_idx > start_idx:
            vmatrix = []
            ematrix = []
            mergematrix = []
            conflict_operators = {}
            oid = 0
            root_idx = []
            for i in range(start_idx, end_idx):
                query = concurrency_df.iloc[i]
                root_idx.append(oid)
                query_runtime = query["run_time_s"]
                (
                    start_time,
                    node_matrix,
                    edge_matrix,
                    conflict_operators,
                    node_merge_matrix,
                    mp_optype,
                    oid,
                ) = extract_plan(
                    parsed_plans[query["query_idx"]],
                    column_table_mapping,
                    all_start_time[i],
                    query_runtime,
                    conflict_operators,
                    all_optype,
                    oid,
                )
                mergematrix = mergematrix + node_merge_matrix
                vmatrix = vmatrix + node_matrix
                ematrix = ematrix + edge_matrix
            ematrix = add_across_plan_relations(conflict_operators, knobs, ematrix)
            root_idx_bit_map = np.zeros(len(vmatrix))
            root_idx_bit_map[root_idx] = 1
            if save_dir:
                np.save(
                    os.path.join(save_dir, f"sample-plan-{run_id}-{wid}-nodes"),
                    np.asarray(vmatrix),
                )
                np.save(
                    os.path.join(save_dir, f"sample-plan-{run_id}-{wid}-edges"),
                    np.asarray(ematrix),
                )
                np.save(
                    os.path.join(save_dir, f"sample-plan-{run_id}-{wid}-merges"),
                    np.asarray(mergematrix),
                )
                np.save(
                    os.path.join(save_dir, f"sample-plan-{run_id}-{wid}-root-idx"),
                    root_idx_bit_map,
                )
            start_idx = end_idx
            wid += 1
        curr_end_time += time_interval


def generate_graph_from_trace(
    data_dir, parsed_queries_path, save_dir, num_operators=20, time_interval=1200
):
    _, all_trace = load_trace_all_version(data_dir, 8, concat=True)
    parsed_run = load_json(parsed_queries_path, namespace=False)
    db = Database("mysql", connect=False)
    knobs = db.fetch_knob()
    all_optype = find_top_k_operators(plans=parsed_run, k=num_operators)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    for run_id, trace in enumerate(all_trace):
        trace = trace.sort_values("time_since_execution_s", ascending=True)
        print(f"generate_graph {run_id}")
        generate_graph(
            trace,
            parsed_run,
            all_optype,
            knobs,
            run_id,
            time_interval=time_interval,
            save_dir=save_dir,
        )
