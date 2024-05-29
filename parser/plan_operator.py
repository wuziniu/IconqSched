# We adapted the legacy from from https://github.com/DataManagementLab/zero-shot-cost-estimation
import math
import re
from typing import List, Optional, Mapping, Any, MutableMapping, Tuple, Set
from parser.utils import child_prod

estimated_regex = re.compile(
    "\(cost=(?P<est_startup_cost>\d+.\d+)..(?P<est_cost>\d+.\d+) rows=(?P<est_card>\d+) width=(?P<est_width>\d+)\)"
)
actual_regex = re.compile(
    "\(actual time=(?P<act_startup_cost>\d+.\d+)..(?P<act_time>\d+.\d+) rows=(?P<act_card>\d+)"
)
op_name_regex = re.compile('->  ([^"(]+)')
workers_planned_regex = re.compile("Workers Planned: (\d+)")
filter_columns_regex = re.compile("([^\(\)\*\+\-'\= ]+)")
literal_regex = re.compile("('[^']+'::[^'\)]+)")


class PlanOperator(dict):
    def __init__(
        self, plain_content: List[str],
            children: Optional[List] = None,
            plan_parameters: Optional[Mapping[str, Any]] = None,
            plan_runtime: float = 0
    ):
        super().__init__()
        self.__dict__ = self
        self.plain_content = plain_content

        self.plan_parameters = (
            plan_parameters if plan_parameters is not None else dict()
        )
        self.children = list(children) if children is not None else []
        self.plan_runtime = plan_runtime

    def parse_lines(
        self, alias_dict: Optional[MutableMapping[str, Optional[str]]] = None
    ) -> None:
        op_line = self.plain_content[0]

        # parse plan operator name
        op_name_match = op_name_regex.search(op_line)
        assert op_name_match is not None
        op_name = op_name_match.groups()[0]
        for split_word in ["on", "using"]:
            if f" {split_word} " in op_name:
                op_name = op_name.split(f" {split_word} ")[0]
        op_name = op_name.strip()

        # operator table
        if " on " in op_line:
            table_name = op_line.split(" on ")[1].strip()
            table_name_parts = table_name.split(" ")

            table_name = table_name_parts[0].strip('"')

            if table_name.endswith("_pkey"):
                table_name = table_name.replace("_pkey", "")

            if "." in table_name:
                table_name = table_name.split(".")[1].strip('"')

            if len(table_name_parts) > 1 and alias_dict is not None:
                potential_alias = table_name_parts[1]
                if potential_alias != "" and not potential_alias.startswith("("):
                    alias_dict[potential_alias] = table_name
                    self.plan_parameters.update(dict(alias=potential_alias))

            if "Subquery Scan" in op_line:
                alias_dict[table_name] = None
            else:
                self.plan_parameters.update(dict(table=table_name))

        self.plan_parameters.update(dict(op_name=op_name))

        # parse estimated plan costs
        match_est = estimated_regex.search(op_line)
        assert match_est is not None
        self.plan_parameters.update(
            {k: float(v) for k, v in match_est.groupdict().items()}
        )

        # parse actual plan costs
        match_act = actual_regex.search(op_line)
        if match_act is not None:
            self.plan_parameters.update(
                {k: float(v) for k, v in match_act.groupdict().items()}
            )

        # collect additional optional information
        for l in self.plain_content[1:]:
            l = l.strip()
            workers_planned_match = workers_planned_regex.search(l)

            if workers_planned_match is not None:
                workers_planned = workers_planned_match.groups()
                if isinstance(workers_planned, list) or isinstance(
                    workers_planned, tuple
                ):
                    workers_planned = workers_planned[0]
                workers_planned = int(workers_planned)
                self.plan_parameters.update(dict(workers_planned=workers_planned))
        self.plain_content = []

    def parse_columns_bottom_up(
        self,
        column_id_mapping: Mapping[Tuple[str, str], int],
        partial_column_name_mapping: Mapping[str, Set[str]],
        table_id_mapping: Mapping[str, int],
        alias_dict: Optional[MutableMapping[str, str]],
    ) -> Set[str]:
        if alias_dict is None:
            alias_dict = dict()

        # first keep track which tables are actually considered here
        node_tables = set()
        if self.plan_parameters.get("table") is not None:
            node_tables.add(self.plan_parameters.get("table"))

        for c in self.children:
            node_tables.update(
                c.parse_columns_bottom_up(
                    column_id_mapping,
                    partial_column_name_mapping,
                    table_id_mapping,
                    alias_dict,
                )
            )

        self.plan_parameters["act_children_card"] = child_prod(self, "act_card")
        self.plan_parameters["est_children_card"] = child_prod(self, "est_card")

        # replace table by id
        table = self.plan_parameters.get("table")
        if table is not None:
            if table in table_id_mapping:
                self.plan_parameters["table"] = table_id_mapping[table]
            else:
                print(f"!!!!!!{self.plan_parameters['table']} not found")
                print(table_id_mapping)
                del self.plan_parameters["table"]

        return node_tables

    def lookup_column_id(
        self, c, column_id_mapping, node_tables, partial_column_name_mapping, alias_dict
    ):
        assert isinstance(c, tuple)
        # here it is clear which column is meant
        if len(c) == 2:
            table = c[0].strip('"')
            column = c[1].strip('"')

            if table in alias_dict:
                table = alias_dict[table]

                # this is a subquery and we cannot uniquely identify the corresponding table
                if table is None:
                    return self.lookup_column_id(
                        (c[1],),
                        column_id_mapping,
                        node_tables,
                        partial_column_name_mapping,
                        alias_dict,
                    )

        # we now have to guess which table this column belongs to
        elif len(c) == 1:
            column = c[0].strip('"')

            potential_tables = partial_column_name_mapping[column].intersection(
                node_tables
            )
            assert len(potential_tables) == 1, (
                f"Did not find unique table for column {column} "
                f"(node_tables: {node_tables})"
            )
            table = list(potential_tables)[0]
        else:
            raise NotImplementedError

        col_id = column_id_mapping[(table, column)]
        return col_id

    def merge_recursively(self, node):
        assert self.plan_parameters["op_name"] == node.plan_parameters["op_name"]
        assert len(self.children) == len(node.children)

        self.plan_parameters.update(node.plan_parameters)
        for self_c, c in zip(self.children, node.children):
            self_c.merge_recursively(c)

    def parse_lines_recursively(
        self, alias_dict: Optional[Mapping[str, str]] = None,
    ):
        self.parse_lines(
            alias_dict=alias_dict,
        )
        for c in self.children:
            c.parse_lines_recursively(
                alias_dict=alias_dict
            )

    def min_card(self):
        act_card = self.plan_parameters.get("act_card")
        if act_card is None:
            act_card = math.inf

        for c in self.children:
            child_min_card = c.min_card()
            if child_min_card < act_card:
                act_card = child_min_card

        return act_card

    def recursive_str(self, pre):
        pre_whitespaces = "".join(["\t" for _ in range(pre)])
        # current_string = '\n'.join([pre_whitespaces + content for content in self.plain_content])
        current_string = pre_whitespaces + str(self.plan_parameters)
        node_strings = [current_string]

        for c in self.children:
            node_strings += c.recursive_str(pre + 1)

        return node_strings

    def __str__(self):
        rec_str = self.recursive_str(0)
        return "\n".join(rec_str)
