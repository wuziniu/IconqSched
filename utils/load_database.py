import psycopg2
import os

from workloads.postgres.imdb_schema import IMDB_SCHEMA, IMDB_LOAD_TEMPLATE, IMDB_TABLE_NAMES
from workloads.postgres.tpc_schema import TPC_SCHEMA, TPC_LOAD_TEMPLATE, TPC_TABLE_NAMES


def load_database_postgres(data_dir: str, db_name: str = "imdb"):
    host = "imdb-postgres.xxx.us-east-1.rds.amazonaws.com"
    port = "5432"
    user = "postgres"
    token = "xxxx"
    conn = psycopg2.connect(host=host, port=port, database="postgres", user=user, password=token)
    conn.autocommit = True
    cur = conn.cursor()
    cur.execute(f"DROP DATABASE IF EXISTS {db_name};")
    cur.execute(f"CREATE DATABASE {db_name};")
    cur.close()
    conn.close()

    if db_name == "imdb":
        schema = IMDB_SCHEMA
        load_template = IMDB_LOAD_TEMPLATE
        table_names = IMDB_TABLE_NAMES
    elif db_name == "tpc":
        schema = TPC_SCHEMA
        load_template = TPC_LOAD_TEMPLATE
        table_names = TPC_TABLE_NAMES
    else:
        assert False, f"unrecognized db_name {db_name}"

    conn = psycopg2.connect(host=host, port=port, database=db_name, user=user, password=token)
    conn.autocommit = True
    cur = conn.cursor()
    cur.execute(schema)

    for table_name in table_names:
        load_query = load_template.format(
            table_name=table_name,
            path=os.path.join(data_dir, f"{table_name}.csv")
        )
        print(load_query)
    conn.commit()
    cur.close()
    conn.close()








