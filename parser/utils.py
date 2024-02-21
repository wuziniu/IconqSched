import re
import json
from types import SimpleNamespace


def canonicalize_join_cond(join_cond, is_brad=False):
    """join_cond: 4-tuple"""
    t1, c1, t2, c2 = join_cond
    if is_brad and not t1.endswith("_brad_source"):
        t1 = t1 + "_brad_source"
    if is_brad and not t2.endswith("_brad_source"):
        t2 = t2 + "_brad_source"
    if t1 < t2:
        return t1, c1, t2, c2
    return t2, c2, t1, c1


def dedup_join_conds(join_conds, is_brad=False):
    """join_conds: list of 4-tuple (t1, c1, t2, c2)."""
    canonical_join_conds = [canonicalize_join_cond(jc, is_brad) for jc in join_conds]
    return sorted(set(canonical_join_conds))


def get_join_conds(sql, is_brad=False):
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


def format_join_cond(tup):
    t1, c1, t2, c2 = tup
    return f"{t1}.{c1} = {t2}.{c2}"


def get_touched_tables(sql):
    all_tables = set()
    join_conds = get_join_conds(sql)
    for t1, c1, t2, c2 in join_conds:
        all_tables.add(t1)
        all_tables.add(t2)
    return all_tables


def load_json(path, namespace=False):
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

