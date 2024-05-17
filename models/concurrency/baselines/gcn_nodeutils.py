import copy


def compute_cost(node):
    node_param = node["plan_parameters"]
    return (
        max(
            (float(node_param["est_cost"]) - float(node_param["est_startup_cost"])), 1.0
        )
        / 1e6
    )


def compute_time(node):
    # in seconds
    node_param = node["plan_parameters"]
    if "act_time" not in node_param:
        return max(float(node_param["est_cost"]) / 1e6, 1e-3)
    else:
        act_time = max(float(node_param["act_time"]), 1.0) / 1e3
        return act_time


def get_table_name_from_filter_columns(node, column_table_mapping, tables):
    if "column" in node and node["column"] is not None:
        table = column_table_mapping[int(node["column"])]
        tables.add(table)
    if "children" in node and node["children"] is not None:
        for child in node["children"]:
            get_table_name_from_filter_columns(child, column_table_mapping, tables)


def get_used_tables(node, column_table_mapping):
    tables = set()
    stack = [node]
    while len(stack) != 0:
        parent = stack.pop(0)
        if "output_columns" in parent:
            for col_feat in parent["output_columns"]:
                for col in col_feat["columns"]:
                    if col is not None:
                        table = column_table_mapping[int(col)]
                        tables.add(table)

        if "filter_columns" in parent and parent["filter_columns"] is not None:
            get_table_name_from_filter_columns(
                parent["filter_columns"], column_table_mapping, tables
            )
        if "Plans" in parent:
            for n in parent["Plans"]:
                stack.append(n)
    return tables


def overlap(node_i, node_j):
    if node_j[1] < node_i[2] and node_i[2] < node_j[2]:
        return (node_i[2] - node_j[1]) / (node_j[2] - min(node_i[1], node_j[1]))
    elif node_i[1] < node_j[2] and node_j[2] < node_i[2]:
        return (node_j[2] - node_i[1]) / (node_i[2] - min(node_i[1], node_j[1]))
    else:
        return 0


def extract_plan(
    sample,
    column_table_mapping,
    start_time,
    query_runtime,
    conflict_operators,
    mp_optype,
    oid,
):
    plan = copy.deepcopy(sample)
    root_id = oid
    node_matrix = []
    edge_matrix = []
    node_merge_matrix = []

    # assign oid for each operator
    stack = [plan]
    while len(stack) != 0:
        curr_node = stack.pop(0)
        curr_node["oid"] = oid
        oid = oid + 1
        if "children" in curr_node:
            for node in curr_node["children"]:
                stack.append(node)

    stack = [plan]
    while len(stack) != 0:
        parent = stack.pop(0)
        run_cost = compute_cost(parent)
        if parent["oid"] == root_id:
            run_time = query_runtime
        else:
            run_time = compute_time(parent)
        # if parent["oid"] == root_id:
        #   print(run_time, root_id)
        node_param = parent["plan_parameters"]
        tables = get_used_tables(node_param, column_table_mapping)
        if "act_startup_cost" in node_param:
            act_startup_cost = node_param["act_startup_cost"] / 1e3
        else:
            act_startup_cost = node_param["est_startup_cost"] / 1e6
        if "act_time" in node_param:
            act_time = node_param["act_time"] / 1e3
        else:
            act_time = node_param["est_cost"] / 1e6
        operator_info = [
            parent["oid"],
            start_time + act_startup_cost,
            start_time + act_time,
        ]

        for table in tables:
            if table not in conflict_operators:
                conflict_operators[table] = [operator_info]
            else:
                conflict_operators[table].append(operator_info)

        op_type = [0 for _ in range(len(mp_optype) + 1)]
        if node_param["op_name"] in mp_optype:
            op_type[mp_optype.index(node_param["op_name"])] = 1
        else:
            op_type[len(mp_optype)] = 1
        node_feature = (
            [parent["oid"]]
            + op_type
            + [
                run_cost,
                float(node_param["est_card"]) / 1e6,
                start_time + float(node_param["est_startup_cost"]) / 1e6,
                run_time,
            ]
        )

        node_matrix.append(node_feature)

        node_merge_feature = (
            [
                parent["oid"],
                start_time + node_param["est_startup_cost"] / 1e6,
                start_time + node_param["est_cost"] / 1e6,
            ]
            + op_type
            + [
                run_cost,
                start_time + float(node_param["est_startup_cost"]) / 1e6,
                run_time,
            ]
        )
        node_merge_matrix.append(node_merge_feature)

        if "children" in parent:
            for node in parent["children"]:
                stack.append(node)
                edge_matrix.append([node["oid"], parent["oid"], 1])

    return (
        start_time,
        node_matrix,
        edge_matrix,
        conflict_operators,
        node_merge_matrix,
        mp_optype,
        oid,
    )


def add_across_plan_relations(conflict_operators, knobs, ematrix):
    data_weight = 0.1
    for knob in knobs:
        data_weight *= knob
    for table in conflict_operators:
        for i in range(len(conflict_operators[table])):
            for j in range(i + 1, len(conflict_operators[table])):
                node_i = conflict_operators[table][i]
                node_j = conflict_operators[table][j]

                time_overlap = overlap(node_i, node_j)
                if time_overlap:
                    ematrix = ematrix + [
                        [node_i[0], node_j[0], -data_weight * time_overlap]
                    ]
                    ematrix = ematrix + [
                        [node_j[0], node_i[0], -data_weight * time_overlap]
                    ]
    return ematrix
