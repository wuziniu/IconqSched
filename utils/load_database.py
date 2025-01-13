import psycopg2
import os

from workloads.postgres.imdb_schema import IMDB_SCHEMA, IMDB_FK_INDEX, IMDB_LOAD_TEMPLATE, IMDB_TABLE_NAMES
from workloads.postgres.tpc_schema import TPC_SCHEMA, TPC_FK_INDEX, TPC_LOAD_TEMPLATE, TPC_TABLE_NAMES
from workloads.redshift.imdb_schema import REDSHIFT_IMDB_SCHEMA, REDSHIFT_IMDB_LOAD_TEMPLATE, REDSHIFT_IMDB_TABLE_NAMES
from workloads.redshift.tpc_schema import REDSHIFT_TPC_SCHEMA, REDSHIFT_TPC_TABLE_NAMES, REDSHIFT_TPC_LOAD_TEMPLATE


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

    # unfortunately, psycopg does not support loading from csv, so you need to execute the \copy statement manually
    print("Manually execute the following in postgres")
    for table_name in table_names:
        load_query = load_template.format(
            table_name=table_name,
            path=os.path.join(data_dir, f"{table_name}.csv")
        )
        print(load_query)
    if db_name == "imdb":
        print(IMDB_FK_INDEX)
    else:
        print(TPC_FK_INDEX)
    conn.commit()
    cur.close()
    conn.close()


def load_database_redshift(s3_path: str, db_name: str = "imdb"):
    # s3_path of format s3://{s3_bucket}/{s3_obj}
    host = "redshift-tpc-h.xxx.us-east-1.rds.amazonaws.com"
    port = "5439"
    user = "awsuser"
    token = "xxxx"
    iam_role = 'arn:aws:iam::xxx:role/RedshiftS3'
    conn = psycopg2.connect(host=host, port=port, database="postgres", user=user, password=token)
    cur = conn.cursor()
    cur.execute(f"DROP DATABASE IF EXISTS {db_name};")
    cur.execute(f"CREATE DATABASE {db_name};")
    cur.close()
    conn.close()

    if db_name == "imdb":
        schema = REDSHIFT_IMDB_SCHEMA
        if type(schema) == list:
            schema = '\n'.join(schema)
        load_template = REDSHIFT_IMDB_LOAD_TEMPLATE
        table_names = REDSHIFT_IMDB_TABLE_NAMES
    elif db_name == "tpc":
        schema = REDSHIFT_TPC_SCHEMA
        load_template = REDSHIFT_TPC_LOAD_TEMPLATE
        table_names = REDSHIFT_TPC_TABLE_NAMES
    else:
        assert False, f"unrecognized db_name {db_name}"

    conn = psycopg2.connect(host=host, port=port, database=db_name, user=user, password=token)
    conn.autocommit = True
    cur = conn.cursor()
    cur.execute(schema)

    for table_name in table_names:
        load_query = load_template.format(
            table_name=table_name,
            s3_path=os.path.join(s3_path, table_name, f"{table_name}.csv"),
            s3_iam_role=iam_role
        )
        cur.execute(load_query)
    conn.commit()
    cur.close()
    conn.close()






