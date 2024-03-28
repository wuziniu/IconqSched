import numpy as np
from parser.utils import load_json


def find_top_k_operators(parsed_queries_path=None, plans=None, k=15, verbose=False):
    if plans is None:
        plans = load_json(parsed_queries_path, namespace=False)
    all_operators = dict()
    for plan in plans['parsed_plans']:
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
    print(f"Top {k} operators contains {explained_ops/np.sum(op_counts)} total operators")
    if verbose:
        for i in idx[:k]:
            print(f"{op_names[i]}: {op_counts[i]/np.sum(op_counts)}")
    return [op_names[i] for i in idx[:k]]


def dfs_all_operators(plan, all_operators):
    if 'plan_parameters' in plan and 'op_name' in plan['plan_parameters']:
        op = plan['plan_parameters']['op_name']
        if op not in all_operators:
            all_operators[op] = 1
        else:
            all_operators[op] += 1
    if 'children' in plan:
        for child in plan['children']:
            dfs_all_operators(child, all_operators)


def dfs_find_operator_size(plan, operators, features, use_size, use_log, true_card):
    if 'plan_parameters' in plan and 'op_name' in plan['plan_parameters']:
        op = plan['plan_parameters']['op_name']
        if op in operators:
            idx = operators.index(op)
            features[idx * 2] += 1
            if true_card:
                card = plan['plan_parameters']['act_card']
            else:
                card = plan['plan_parameters']['est_card']
            if use_size:
                card = card * plan['plan_parameters']['est_width']
            if use_log:
                features[idx * 2 + 1] += max(np.log(card + 1e-5), 0)
            else:
                features[idx * 2 + 1] += (card/1024/1024)  # convert byte to mb
    if 'children' in plan:
        for child in plan['children']:
            dfs_find_operator_size(child, operators, features, use_size, use_log, true_card)


def featurize_one_plan(plan, operators, use_size=False, use_log=True, true_card=False):
    features = np.zeros(len(operators) * 2)
    dfs_find_operator_size(plan, operators, features, use_size, use_log, true_card)
    return features
