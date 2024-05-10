import json
import configparser
import psycopg2
import pymysql
import pymysql.cursors as pycursor
import time
import numpy as np
from models.concurrency.baselines.gcn_constant import CONFIGPATH


class DictParser(configparser.ConfigParser):
    def read_dict(self):
        d = dict(self._sections)
        for k in d:
            d[k] = dict(d[k])
        return d


"""
cf = DictParser()
cf.read("config.ini", encoding="utf-8")
config_dict = cf.read_dict()
"""


def load_config():
    cf = DictParser()
    cf.read(CONFIGPATH, encoding="utf-8")
    config_dict = cf.read_dict()
    return config_dict


class Database:
    def __init__(
        self, server_name="postgresql", config_dict=load_config(), connect=True
    ):
        def parse_knob_config(config_dict):
            _knob_config = config_dict["knob_config"]
            for key in _knob_config:
                _knob_config[key] = json.loads(str(_knob_config[key]).replace("'", '"'))
            return _knob_config

        knob_config = parse_knob_config(config_dict)
        self.knob_names = list(knob_config.keys())
        self.knob_config = knob_config
        self.server_name = server_name
        self.connect = connect
        # print("knob_names:", self.knob_names)
        if connect:
            try:
                conn = self._get_conn()
                cursor = conn.cursor()
                sql = "SELECT count FROM INFORMATION_SCHEMA.INNODB_METRICS where status='enabled'"
                cursor.execute(sql)
                result = cursor.fetchall()

                self.internal_metric_num = len(result)
                cursor.close()
                conn.close()
            except Exception as err:
                print("execute sql error:", err)

    def _get_conn(self):
        # TODO: change the hard-coded connections here
        if self.server_name == "mysql":
            sucess = 0
            conn = -1
            count = 0
            while not sucess and count < 3:
                try:
                    conn = pymysql.connect(
                        host="166.111.121.62",
                        port=3306,
                        user="feng",
                        password="db10204",
                        db="INFORMATION_SCHEMA",
                        connect_timeout=36000,
                        cursorclass=pycursor.DictCursor,
                    )

                    sucess = 1
                except Exception as result:
                    count += 1
                    time.sleep(10)
            if conn == -1:
                raise Exception

            return conn

        elif self.server_name == "postgresql":
            sucess = 0
            conn = -1
            count = 0
            while not sucess and count < 3:
                try:
                    db_name = "INFORMATION_SCHEMA"  # zxn Modified.
                    conn = psycopg2.connect(
                        database="INFORMATION_SCHEMA",
                        user="lixizhang",
                        password="xi10261026zhang",
                        host="166.111.5.177",
                        port="5433",
                    )
                    sucess = 1
                except Exception as result:
                    count += 1
                    time.sleep(10)
            if conn == -1:
                raise Exception
            return conn

        else:
            print("Cannot connect to database...")
            return

    def fetch_knob(self):
        state_list = []
        if self.connect:
            try:
                conn = self._get_conn()
                cursor = conn.cursor()
                sql = "select"
                for i, knob in enumerate(self.knob_names):
                    sql = sql + " @@" + knob

                    if i < len(self.knob_names) - 1:
                        sql = sql + ", "

                # state metrics
                cursor.execute(sql)
                result = cursor.fetchall()

                for i in range(len(self.knob_names)):
                    value = (
                        result[0]["@@%s" % self.knob_names[i]]
                        if result[0]["@@%s" % self.knob_names[i]] != 0
                        else self.knob_config[self.knob_names[i]]["max_value"]
                    )  # not limit if value equals 0

                    # print(value, self.knob_config[self.knob_names[i]]["max_value"], self.knob_config[self.knob_names[i]]["min_value"])
                    state_list.append(
                        value
                        / (
                            self.knob_config[self.knob_names[i]]["max_value"]
                            - self.knob_config[self.knob_names[i]]["min_value"]
                        )
                    )
                cursor.close()
                conn.close()
            except Exception as error:
                print("fetch_knob Error:", error)
        else:
            for i in range(len(self.knob_names)):
                max_value = self.knob_config[self.knob_names[i]]["max_value"]
                min_value = self.knob_config[self.knob_names[i]]["min_value"]
                state_list.append(max_value / (max_value - min_value))
        state_list = np.asarray(state_list)
        return state_list
