import numpy as np
from parser.utils import load_json


def get_top_k_table_by_size(parsed_queries_path=None, plans=None, k=15, use_name=False):
    if plans is None:
        plans = load_json(parsed_queries_path, namespace=False)
    table_stats = plans["database_stats"]["table_stats"]
    table_sizes = [tab["reltuples"] for tab in table_stats]
    selected_table_sizes = dict()
    idx = np.argsort(table_sizes)[::-1]
    k = min(k, len(table_stats))
    for i in idx[:k]:
        if use_name:
            tab_name = table_stats[i]["relname"]
        else:
            tab_name = i
        selected_table_sizes[tab_name] = table_sizes[i]
    return selected_table_sizes


def find_top_k_operators(parsed_queries_path=None, plans=None, k=15, verbose=False):
    if plans is None:
        plans = load_json(parsed_queries_path, namespace=False)
    all_operators = dict()
    for plan in plans["parsed_plans"]:
        dfs_all_operators(plan, all_operators)
    op_names = []
    op_counts = []
    for op in all_operators:
        op_names.append(op)
        op_counts.append(all_operators[op])
    op_counts = np.asarray(op_counts)
    idx = np.argsort(op_counts)[::-1]
    k = min(k, len(op_names))
    explained_ops = np.sum(op_counts[idx][:k])
    print(
        f"Top {k} operators contains {explained_ops/np.sum(op_counts)} total operators"
    )
    if verbose:
        for i in idx[:k]:
            print(f"{op_names[i]}: {op_counts[i]/np.sum(op_counts)}")
    return [op_names[i] for i in idx[:k]]


def dfs_all_operators(plan, all_operators):
    if "plan_parameters" in plan and "op_name" in plan["plan_parameters"]:
        op = plan["plan_parameters"]["op_name"]
        if op not in all_operators:
            all_operators[op] = 1
        else:
            all_operators[op] += 1
    if "children" in plan:
        for child in plan["children"]:
            dfs_all_operators(child, all_operators)


def dfs_find_operator_size(
    plan,
    operators,
    all_table_size,
    features,
    table_features,
    use_size,
    use_log,
    true_card,
    use_table_selectivity,
    return_cardinalities
):
    if "plan_parameters" in plan and "op_name" in plan["plan_parameters"]:
        op = plan["plan_parameters"]["op_name"]
        if op in operators:
            idx = operators.index(op)
            features[idx * 2] += 1
            if true_card:
                if "act_card" not in plan["plan_parameters"]:
                    card = plan["plan_parameters"]["est_card"]
                else:
                    card = plan["plan_parameters"]["act_card"]
            else:
                card = plan["plan_parameters"]["est_card"]
            if use_size:
                card = card * plan["plan_parameters"]["est_width"]
            return_cardinalities.append(card / 1024 / 1024)
            if use_log:
                features[idx * 2 + 1] += max(np.log(card + 1e-5), 0)
            else:
                features[idx * 2 + 1] += card / 1024 / 1024  # convert byte to mb
            if table_features is not None:
                if "table" in plan["plan_parameters"]:
                    table_name = plan["plan_parameters"]["table"]
                    if table_name in all_table_size:
                        table_idx = list(all_table_size.keys()).index(table_name)
                        if use_table_selectivity:
                            if true_card:
                                if "act_card" not in plan["plan_parameters"]:
                                    card = plan["plan_parameters"]["est_card"]
                                else:
                                    card = plan["plan_parameters"]["act_card"]
                            else:
                                card = plan["plan_parameters"]["est_card"]
                            table_features[table_idx] = (
                                card / all_table_size[table_name]
                            )
                        else:
                            if use_log:
                                table_features[table_idx] = max(np.log(card + 1e-5), 0)
                            else:
                                table_features[table_idx] = card / 1024 / 1024
    if "children" in plan:
        for child in plan["children"]:
            dfs_find_operator_size(
                child,
                operators,
                all_table_size,
                features,
                table_features,
                use_size,
                use_log,
                true_card,
                use_table_selectivity,
                return_cardinalities
            )


def featurize_one_plan(
    plan,
    operators,
    all_table_size=None,
    use_size=False,
    use_log=True,
    true_card=False,
    use_table_selectivity=False,
    return_memory_est=False
):
    features = np.zeros(len(operators) * 2)
    if all_table_size is not None:
        table_features = np.zeros(len(all_table_size))
    else:
        table_features = None
    return_cardinalities = []
    dfs_find_operator_size(
        plan,
        operators,
        all_table_size,
        features,
        table_features,
        use_size,
        use_log,
        true_card,
        use_table_selectivity,
        return_cardinalities
    )
    if table_features is not None:
        features = np.concatenate((features, table_features))
    if return_memory_est:
        return features, np.max(return_cardinalities)
    else:
        return features
