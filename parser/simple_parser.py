import os
import numpy as np
from parser.utils import get_join_conds, get_touched_tables, load_json


def get_table_feature(sql, all_table_list):
    feature = np.zeros(len(all_table_list))
    all_tables = get_touched_tables(sql)
    for table in all_tables:
        i = all_table_list.index(table)
        feature[i] = 1
    return feature
