from typing import List


TPC_SCHEMA: str = """
CREATE TABLE region  ( r_regionkey  INTEGER NOT NULL,
                            r_name       CHAR(25) NOT NULL,
                            r_comment    VARCHAR(152),
                            PRIMARY KEY (r_regionkey)
                            );
                            
CREATE TABLE nation  ( n_nationkey  INTEGER NOT NULL,
                            n_name       CHAR(25) NOT NULL,
                            n_regionkey  INTEGER NOT NULL,
                            n_comment    VARCHAR(152),
                            PRIMARY KEY (n_nationkey)
                            );

CREATE TABLE part  ( p_partkey     INTEGER NOT NULL,
                          p_name        VARCHAR(55) NOT NULL,
                          p_mfgr        CHAR(25) NOT NULL,
                          p_brand       CHAR(10) NOT NULL,
                          p_type        VARCHAR(25) NOT NULL,
                          p_size        INTEGER NOT NULL,
                          p_container   CHAR(10) NOT NULL,
                          p_retailprice DECIMAL(15,2) NOT NULL,
                          p_comment     VARCHAR(23) NOT NULL,
                          PRIMARY KEY (p_partkey)
                           );

CREATE TABLE supplier ( s_suppkey     INTEGER NOT NULL,
                             s_name        CHAR(25) NOT NULL,
                             s_address     VARCHAR(40) NOT NULL,
                             s_nationkey   INTEGER NOT NULL,
                             s_phone       CHAR(15) NOT NULL,
                             s_acctbal     DECIMAL(15,2) NOT NULL,
                             s_comment     VARCHAR(101) NOT NULL,
                             PRIMARY KEY (s_suppkey)
                             );

CREATE TABLE partsupp ( ps_partkey     INTEGER NOT NULL,
                             ps_suppkey     INTEGER NOT NULL,
                             ps_availqty    INTEGER NOT NULL,
                             ps_supplycost  DECIMAL(15,2)  NOT NULL,
                             ps_comment     VARCHAR(199) NOT NULL,
                             PRIMARY KEY (ps_partkey, ps_suppkey)
                            );

CREATE TABLE customer ( c_custkey     INTEGER NOT NULL,
                             c_name        VARCHAR(25) NOT NULL,
                             c_address     VARCHAR(40) NOT NULL,
                             c_nationkey   INTEGER NOT NULL,
                             c_phone       CHAR(15) NOT NULL,
                             c_acctbal     DECIMAL(15,2)   NOT NULL,
                             c_mktsegment  CHAR(10) NOT NULL,
                             c_comment     VARCHAR(117) NOT NULL,
                             PRIMARY KEY (c_custkey)
                             );

CREATE TABLE orders  ( o_orderkey       INTEGER NOT NULL,
                           o_custkey        INTEGER NOT NULL,
                           o_orderstatus    CHAR(1) NOT NULL,
                           o_totalprice     DECIMAL(15,2) NOT NULL,
                           o_orderdate      DATE NOT NULL,
                           o_orderpriority  CHAR(15) NOT NULL,  
                           o_clerk          CHAR(15) NOT NULL, 
                           o_shippriority   INTEGER NOT NULL,
                           o_comment        VARCHAR(79) NOT NULL,
                           PRIMARY KEY (o_orderkey)
                           );

CREATE TABLE lineitem ( l_orderkey    INTEGER NOT NULL,
                             l_partkey     INTEGER NOT NULL,
                             l_suppkey     INTEGER NOT NULL,
                             l_linenumber  INTEGER NOT NULL,
                             l_quantity    DECIMAL(15,2) NOT NULL,
                             l_extendedprice  DECIMAL(15,2) NOT NULL,
                             l_discount    DECIMAL(15,2) NOT NULL,
                             l_tax         DECIMAL(15,2) NOT NULL,
                             l_returnflag  CHAR(1) NOT NULL,
                             l_linestatus  CHAR(1) NOT NULL,
                             l_shipdate    DATE NOT NULL,
                             l_commitdate  DATE NOT NULL,
                             l_receiptdate DATE NOT NULL,
                             l_shipinstruct CHAR(25) NOT NULL,
                             l_shipmode     CHAR(10) NOT NULL,
                             l_comment      VARCHAR(44) NOT NULL,
                             PRIMARY KEY (l_orderkey, l_linenumber)
                             );                             
"""

TPC_FK_INDEX: str = """
ALTER TABLE nation ADD CONSTRAINT nation_fk1 FOREIGN KEY (n_regionkey) REFERENCES region (r_regionkey) NOT VALID;
ALTER TABLE supplier ADD CONSTRAINT supplier_fk1 FOREIGN KEY (s_nationkey) references nation (n_nationkey) NOT VALID;
ALTER TABLE partsupp ADD CONSTRAINT partsupp_fk1 FOREIGN KEY (ps_suppkey) references supplier (s_suppkey) NOT VALID;
ALTER TABLE partsupp ADD CONSTRAINT partsupp_fk2 FOREIGN KEY (ps_partkey) references part (p_partkey) NOT VALID;
ALTER TABLE customer ADD CONSTRAINT customer_fk1 FOREIGN KEY (c_nationkey) references nation (n_nationkey) NOT VALID;
ALTER TABLE orders ADD CONSTRAINT orders_fk1 FOREIGN KEY (o_custkey) references customer (c_custkey) NOT VALID;
ALTER TABLE lineitem ADD CONSTRAINT lineitem_fk1 FOREIGN KEY (l_orderkey) references orders (o_orderkey) NOT VALID;
ALTER TABLE lineitem ADD CONSTRAINT lineitem_fk2 FOREIGN KEY (l_partkey, l_suppkey) references partsupp (ps_partkey, ps_suppkey) NOT VALID;

CREATE INDEX idx_nation_fk1 ON nation (n_regionkey);
CREATE INDEX idx_supplier_fk1 ON supplier (s_nationkey);
CREATE INDEX idx_partsupp_fk1 ON partsupp (ps_suppkey);
CREATE INDEX idx_partsupp_fk2 ON partsupp (ps_partkey);
CREATE INDEX idx_customer_fk1 ON customer (c_nationkey);
CREATE INDEX idx_orders_fk1 ON orders (o_custkey);
CREATE INDEX idx_lineitem_fk1 ON lineitem (l_orderkey);
CREATE INDEX idx_lineitem_fk2 ON lineitem (l_partkey, l_suppkey);
"""

TPC_TABLE_NAMES: List[str] = ["nation", "region", "part", "supplier", "partsupp", "customer", "orders", "lineitem"]

TPC_LOAD_TEMPLATE: str = "COPY {table_name} FROM '{path}' with (FORMAT csv, DELIMITER '|');"
