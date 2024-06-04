import re
import json
import numpy as np
from types import SimpleNamespace
from typing import Tuple, List, Set, Mapping, Any, Union


def canonicalize_join_cond(
    join_cond: Tuple[str, str, str, str], is_brad: bool = False
) -> Tuple[str, str, str, str]:
    """join_cond: 4-tuple"""
    t1, c1, t2, c2 = join_cond
    if is_brad and not t1.endswith("_brad_source"):
        t1 = t1 + "_brad_source"
    if is_brad and not t2.endswith("_brad_source"):
        t2 = t2 + "_brad_source"
    if t1 < t2:
        return t1, c1, t2, c2
    return t2, c2, t1, c1


def dedup_join_conds(
    join_conds: List[Tuple[str, str, str, str]], is_brad: bool = False
) -> List[Tuple[str, str, str, str]]:
    """join_conds: list of 4-tuple (t1, c1, t2, c2)."""
    canonical_join_conds = [canonicalize_join_cond(jc, is_brad) for jc in join_conds]
    return sorted(set(canonical_join_conds))


def get_join_conds(sql: str, is_brad: bool = False) -> List[Tuple[str, str, str, str]]:
    """Returns a list of join conditions in the form of (t1, c1, t2, c2)."""
    join_cond_pat = re.compile(
        r"""
        (\w+)  # 1st table
        \.     # the dot "."
        (\w+)  # 1st table column
        \s*    # optional whitespace
        =      # the equal sign "="
        \s*    # optional whitespace
        (\w+)  # 2nd table
        \.     # the dot "."
        (\w+)  # 2nd table column
        """,
        re.VERBOSE,
    )
    join_conds = join_cond_pat.findall(sql)
    if len(join_conds) == 0:
        join_cond_pat = re.compile(
            r"""
            \"
            (\w+)  # 1st table
            \"
            \.     # the dot "."
            \"
            (\w+)  # 1st table column
            \"
            \s*    # optional whitespace
            =      # the equal sign "="
            \s*    # optional whitespace
            \"
            (\w+)  # 2nd table
            \"
            \.     # the dot "."
            \"
            (\w+)  # 2nd table column
            \"
            """,
            re.VERBOSE,
        )
        join_conds = join_cond_pat.findall(sql)
        return dedup_join_conds(join_conds, is_brad)
    return dedup_join_conds(join_conds, is_brad)


def format_join_cond(tup: Tuple[str, str, str, str]) -> str:
    t1, c1, t2, c2 = tup
    return f"{t1}.{c1} = {t2}.{c2}"


def get_touched_tables(sql: str) -> Set[str]:
    all_tables = set()
    join_conds = get_join_conds(sql)
    for t1, c1, t2, c2 in join_conds:
        all_tables.add(t1)
        all_tables.add(t2)
    return all_tables


def dfs_cardinality(
    plan_node: Mapping[str, Any],
    result: List[int],
    with_width: bool = False,
    use_true_card: bool = False,
) -> None:
    if "plan_parameters" in plan_node:
        if (
            "est_card" in plan_node["plan_parameters"]
            and "est_width" in plan_node["plan_parameters"]
        ):
            if use_true_card and "act_card" in plan_node["plan_parameters"]:
                card = plan_node["plan_parameters"]["act_card"]
            else:
                card = plan_node["plan_parameters"]["est_card"]
            if with_width:
                result.append(card * plan_node["plan_parameters"]["est_width"])
            else:
                result.append(card)
    if "children" in plan_node:
        for child in plan_node["children"]:
            dfs_cardinality(child, result, with_width, use_true_card)


def estimate_scan_in_mb(
    db_stats: Mapping[str, Any],
    parsed_query: Mapping[str, Any],
    use_true_card: bool = False,
    is_column_store: bool = False,
) -> Tuple[float, Mapping[str, float]]:
    # TODO: taking the index info and filter info into consideration
    column_stats = db_stats["column_stats"]
    est_scan = 0
    est_scan_per_table = dict()
    for table in parsed_query["scan_nodes"]:
        table_idx = parsed_query["scan_nodes"][table]["table"]
        scan_info = parsed_query["scan_nodes"][table]["plan_parameters"]
        if use_true_card and "act_card" in scan_info:
            card = scan_info["act_card"]
        else:
            card = scan_info["est_card"]
        column_width = 0
        if is_column_store:
            if parsed_query["scan_nodes"][table]["output_columns"] is not None:
                for column in parsed_query["scan_nodes"][table]["output_columns"]:
                    for column_idx in column["columns"]:
                        column_width += column_stats[column_idx]["avg_width"]
        else:
            for column in column_stats:
                if column["tablename"] == db_stats["table_stats"][table_idx]["relname"]:
                    column_width += column["avg_width"]
        scan_in_mb = card * column_width / (1024 * 1024)
        est_scan_per_table[table_idx] = scan_in_mb
        est_scan += scan_in_mb
    return est_scan, est_scan_per_table


def load_json(path: str, namespace: bool = False) -> Union[Mapping, SimpleNamespace]:
    with open(path) as json_file:
        if namespace:
            json_obj = json.load(json_file, object_hook=lambda d: SimpleNamespace(**d))
        else:
            json_obj = json.load(json_file)
    return json_obj


def dumper(obj):
    try:
        return obj.toJSON()
    except:
        return obj.__dict__


def list_columns(n, columns):
    columns.add((n.column, n.operator))
    for c in n.children:
        list_columns(c, columns)


def plan_statistics(
    plan_op,
    tables=None,
    filter_columns=None,
    operators=None,
    skip_columns=False,
    conv_to_dict=False,
):
    if tables is None:
        tables = set()
    if operators is None:
        operators = set()
    if filter_columns is None:
        filter_columns = set()

    params = plan_op.plan_parameters

    if conv_to_dict:
        params = vars(params)

    if "table" in params:
        tables.add(params["table"])
    if "op_name" in params:
        operators.add(params["op_name"])
    if "filter_columns" in params and not skip_columns:
        list_columns(params["filter_columns"], filter_columns)

    for c in plan_op.children:
        plan_statistics(
            c,
            tables=tables,
            filter_columns=filter_columns,
            operators=operators,
            skip_columns=skip_columns,
            conv_to_dict=conv_to_dict,
        )

    return tables, filter_columns, operators


def child_prod(p, feature_name, default=1):
    child_feat = [
        c.plan_parameters.get(feature_name)
        for c in p.children
        if c.plan_parameters.get(feature_name) is not None
    ]
    if len(child_feat) == 0:
        return default
    return np.prod(child_feat)
