# We adapted the legacy from from https://github.com/DataManagementLab/zero-shot-cost-estimation
import collections
import json
import psycopg
import re
from types import SimpleNamespace
import numpy as np
from tqdm import tqdm
from typing import Optional, List, Union, Tuple, Mapping, Set, Dict
from parser.plan_operator import (
    PlanOperator,
)
from parser.utils import plan_statistics, dumper


planning_time_regex = re.compile("planning time: (?P<planning_time>\d+.\d+) ms")
ex_time_regex = re.compile("execution time: (?P<execution_time>\d+.\d+) ms")
init_plan_regex = re.compile("InitPlan \d+ \(returns \$\d\)")
join_columns_regex = re.compile("\w+\.\w+ ?= ?\w+\.\w+")


def create_node(
    lines_plan_operator: List[str], operators_current_level: List[PlanOperator]
) -> List[str]:
    if len(lines_plan_operator) > 0:
        last_operator = PlanOperator(lines_plan_operator)
        operators_current_level.append(last_operator)
        lines_plan_operator = []
    return lines_plan_operator


def count_left_whitespaces(a: str) -> int:
    return len(a) - len(a.lstrip(" "))


def parse_recursively(
    parent: Optional[PlanOperator], plan: List[str], offset: int, depth: int
) -> Union[int, PlanOperator]:
    lines_plan_operator = []
    i = offset
    operators_current_level = []
    while i < len(plan):
        # new operator
        if plan[i].strip().startswith("->"):
            # create plan node for previous one
            lines_plan_operator = create_node(
                lines_plan_operator, operators_current_level
            )

            # if plan operator is deeper
            new_depth = count_left_whitespaces(plan[i])
            if new_depth > depth:
                assert len(operators_current_level) > 0, "No parent found at this level"
                i = parse_recursively(operators_current_level[-1], plan, i, new_depth)

            # one step up in recursion
            elif new_depth < depth:
                break

            # new operator in current depth
            elif new_depth == depth:
                lines_plan_operator.append(plan[i])
                i += 1

        else:
            lines_plan_operator.append(plan[i])
            i += 1

    create_node(lines_plan_operator, operators_current_level)

    # any node in the recursion
    if parent is not None:
        parent.children = operators_current_level
        return i

    # top node
    else:
        # there should only be one top node
        assert len(operators_current_level) == 1
        return operators_current_level[0]


def parse_one_plan(
    analyze_plan_tuples: List, analyze: bool = True, parse: bool = True
) -> Tuple[PlanOperator, float, float]:
    plan_steps = analyze_plan_tuples
    if isinstance(analyze_plan_tuples[0], tuple) or isinstance(
        analyze_plan_tuples[0], list
    ):
        plan_steps = [t[0] for t in analyze_plan_tuples]

    # for some reason this is missing in postgres
    # in order to parse this, we add it
    plan_steps[0] = "->  " + plan_steps[0]

    ex_time = 0
    planning_time = 0
    planning_idx = -1
    if analyze:
        for i, plan_step in enumerate(plan_steps):
            plan_step = plan_step.lower()
            ex_time_match = planning_time_regex.match(plan_step)
            if ex_time_match is not None:
                planning_idx = i
                planning_time = float(ex_time_match.groups()[0])

            ex_time_match = ex_time_regex.match(plan_step)
            if ex_time_match is not None:
                ex_time = float(ex_time_match.groups()[0])

        assert ex_time != 0 and planning_time != 0
        plan_steps = plan_steps[:planning_idx]

    root_operator = None
    if parse:
        root_operator = parse_recursively(None, plan_steps, 0, 0)

    return root_operator, ex_time, planning_time


def parse_plans(
    run_stats: SimpleNamespace,
    explain_only: bool = False,
):
    # keep track of column statistics
    column_id_mapping = dict()
    table_id_mapping = dict()
    partial_column_name_mapping = collections.defaultdict(set)

    database_stats = run_stats.database_stats
    # enrich column stats with table sizes
    table_sizes = dict()
    for table_stat in database_stats.table_stats:
        table_sizes[table_stat.relname] = table_stat.reltuples

    for i, column_stat in enumerate(database_stats.column_stats):
        table = column_stat.tablename
        column = column_stat.attname
        column_stat.table_size = table_sizes[table]
        column_id_mapping[(table, column)] = i
        partial_column_name_mapping[column].add(table)

    # similar for table statistics
    for i, table_stat in enumerate(database_stats.table_stats):
        table = table_stat.relname
        table_id_mapping[table] = i

    # parse individual queries
    parsed_plans = []
    sql_queries = []
    avg_runtimes = []
    no_tables = []
    no_filters = []
    op_perc = collections.defaultdict(int)
    for query_no, q in enumerate(tqdm(run_stats.query_list)):
        curr_explain_only = explain_only
        alias_dict = dict()
        analyze_plan = None
        if not curr_explain_only and not (hasattr(q, "timeout") and q.timeout):
            if q.analyze_plans is None:
                continue

            if len(q.analyze_plans) == 0:
                continue

            # subqueries are currently not supported
            analyze_str = "".join([l[0] for l in q.verbose_plan])
            if "SubPlan" in analyze_str or "InitPlan" in analyze_str:
                continue

            # check if it just initializes a plan
            if isinstance(q.analyze_plans[0][0], list):
                analyze_plan_string = "".join(l[0] for l in q.analyze_plans[0])
            else:
                analyze_plan_string = "".join(q.analyze_plans)
            if init_plan_regex.search(analyze_plan_string) is not None:
                continue

            # compute average execution and planning times
            ex_times = []
            planning_times = []
            for analyze_plan in q.analyze_plans:
                _, ex_time, planning_time = parse_one_plan(
                    analyze_plan, analyze=True, parse=False
                )
                ex_times.append(ex_time)
                planning_times.append(planning_time)
            avg_runtime = sum(ex_times) / len(ex_times)

            # parse the plan as a tree
            analyze_plan, _, _ = parse_one_plan(
                q.analyze_plans[0], analyze=True, parse=True
            )

            # parse information contained in operator nodes (different information in verbose and analyze plan)
            analyze_plan.parse_lines_recursively(
                alias_dict=alias_dict,
            )

        # elif timeout:
        #     avg_runtime = float(2 * max_runtime)

        else:
            avg_runtime = 0

        # only explain plan (not executed)
        verbose_plan, _, _ = parse_one_plan(q.verbose_plan, analyze=False, parse=True)
        verbose_plan.parse_lines_recursively(
            alias_dict=alias_dict,
        )

        if analyze_plan is not None:
            # merge the plans with different information
            analyze_plan.merge_recursively(verbose_plan)

        else:
            analyze_plan = verbose_plan

        tables, filter_columns, operators = plan_statistics(analyze_plan)

        analyze_plan.parse_columns_bottom_up(
            column_id_mapping,
            partial_column_name_mapping,
            table_id_mapping,
            alias_dict=alias_dict,
        )
        analyze_plan.tables = tables
        analyze_plan.num_tables = len(tables)
        analyze_plan.plan_runtime = avg_runtime

        # collect statistics
        avg_runtimes.append(avg_runtime)
        no_tables.append(len(tables))
        for _, op in filter_columns:
            op_perc[op] += 1
        # log number of filters without counting AND, OR
        no_filters.append(len([fc for fc in filter_columns if fc[0] is not None]))

        parsed_plans.append(analyze_plan)
        sql_queries.append(q.sql)

    # statistics in seconds
    print(
        f"Table statistics: "
        f"\n\tmean: {np.mean(no_tables):.1f}"
        f"\n\tmedian: {np.median(no_tables)}"
        f"\n\tmax: {np.max(no_tables)}"
    )
    print("Operators statistics (appear in x% of queries)")
    for op, op_count in op_perc.items():
        print(f"\t{str(op)}: {op_count / len(avg_runtimes) * 100:.0f}%")
    print(
        f"Runtime statistics: "
        f"\n\tmedian: {np.median(avg_runtimes) / 1000:.2f}s"
        f"\n\tmax: {np.max(avg_runtimes) / 1000:.2f}s"
        f"\n\tmean: {np.mean(avg_runtimes) / 1000:.2f}s"
    )
    print(
        f"Parsed {len(parsed_plans)} plans ({len(run_stats.query_list) - len(parsed_plans)} had zero-cardinalities "
        f"or were too fast)."
    )

    parsed_runs = dict(
        parsed_plans=parsed_plans,
        sql_queries=sql_queries,
        database_stats=database_stats,
        run_kwargs=run_stats.run_kwargs,
    )
    stats = dict(
        runtimes=str(avg_runtimes), no_tables=str(no_tables), no_filters=str(no_filters)
    )
    return parsed_runs, stats


def parse_one_plan_online(
    sql: str,
    column_id_mapping: Mapping[Tuple[str, str], int],
    partial_column_name_mapping: Mapping[str, Set[str]],
    table_id_mapping: Mapping[str, int],
    db_conn: psycopg.connection,
) -> PlanOperator:
    with db_conn.cursor() as cur:
        cur.execute("EXPLAIN VERBOSE " + sql)
        verbose_plan = cur.fetchall()

    verbose_plan, _, _ = parse_one_plan(verbose_plan, analyze=False, parse=True)
    alias_dict = dict()
    verbose_plan.parse_lines_recursively(
        alias_dict=alias_dict,
    )
    tables, _, _ = plan_statistics(verbose_plan)
    verbose_plan.parse_columns_bottom_up(
        column_id_mapping,
        partial_column_name_mapping,
        table_id_mapping,
        alias_dict=alias_dict,
    )
    verbose_plan.tables = tables
    verbose_plan.num_tables = len(tables)
    return verbose_plan


def transform_dicts(column_stats_names: List[str], column_stats_rows: List) -> List:
    return [
        {k: v for k, v in zip(column_stats_names, row)} for row in column_stats_rows
    ]


def get_query_plans(query_file: str,
                    db_conn: psycopg.connection,
                    save_file: Optional[str] = None):
    database_stats: Dict = dict()
    column_stats_query = """
                SELECT s.tablename, s.attname, s.null_frac, s.avg_width, s.n_distinct, s.correlation, c.data_type 
                FROM pg_stats s
                JOIN information_schema.columns c ON s.tablename=c.table_name AND s.attname=c.column_name
                WHERE s.schemaname='public';
            """
    with db_conn.cursor() as cur:
        cur.execute(column_stats_query)
        column_stats_rows = cur.fetchall()
        column_stats_names = [desc[0] for desc in cur.description]
    column_stats = transform_dicts(column_stats_names, column_stats_rows)
    database_stats['column_stats'] = column_stats

    table_stats_query = """SELECT relname, reltuples, relpages from pg_class 
                           WHERE relkind = 'r' and relname NOT LIKE 'pg_%' 
                           and relname NOT LIKE 'sql_%';"""
    with db_conn.cursor() as cur:
        cur.execute(table_stats_query)
        table_stats_rows = cur.fetchall()
        table_stats_names = [desc[0] for desc in cur.description]
    table_stats = transform_dicts(table_stats_names, table_stats_rows)
    for table in table_stats:
        if table['relname'] == 'region':
            # for some reason, region has -1 tuples as in pg_class
            table['reltuples'] = 5.0
            table['relpages'] = 1
    database_stats['table_stats'] = table_stats

    with open(query_file, "r") as f:
        queries_text = f.read()
    queries = queries_text.split(";")[:-1]
    queries = [q.strip() + ";" for q in queries]

    column_id_mapping: Dict[Tuple[str, str], int] = dict()
    partial_column_name_mapping: Dict[str, Set[str]] = collections.defaultdict(set)
    table_id_mapping: Dict[str, int] = dict()

    for i, column_stat in enumerate(column_stats):
        table = column_stat["tablename"]
        column = column_stat["attname"]
        column_id_mapping[(table, column)] = i
        partial_column_name_mapping[column].add(table)

    for i, table_stat in enumerate(table_stats):
        table = table_stat["relname"]
        table_id_mapping[table] = i

    parsed_plans: List = []
    for query in queries:
        verbose_plan = parse_one_plan_online(query,
                                             column_id_mapping,
                                             partial_column_name_mapping,
                                             table_id_mapping,
                                             db_conn)
        parsed_plans.append(verbose_plan)

    parsed_queries = dict(database_stats=database_stats, parsed_plans=parsed_plans)
    if save_file:
        with open(save_file, "w") as outfile:
            json.dump(parsed_queries, outfile, default=dumper)
    return parsed_queries
