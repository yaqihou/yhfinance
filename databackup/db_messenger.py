
import os
import sqlite3

import pandas as pd
import datetime as dt

from .defs import TableName

DB_NAME = os.path.join(
    os.environ.get('HOME', '.'), 'Dropbox', '66-DBs', 'FinDB.db'
)

class DBMessenger:

    _conn_dict: dict[str, sqlite3.Connection] = {}

    def __init__(self, db_name: str = DB_NAME):
        self._db_name = db_name

        self._on_init_check_meta_table()

    @property
    def conn(self) -> sqlite3.Connection:
        if self._db_name not in self._conn_dict:
            self._conn_dict[self._db_name] = sqlite3.connect(self._db_name)
        return self._conn_dict[self._db_name]

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        return
    
    def read_sql(self, sql: str) -> pd.DataFrame:
        # TODO - add exception treatment
        return pd.read_sql(sql, self.conn)

    # TODO - def maintain_tickers_meta

    # TODO - def maintain_tasks_meta

    # TODO - def maintain_unique_entries

    def _exist_table(self, tbl_name):
        with self.conn as conn:
            cur = conn.cursor()
            res = cur.execute(f"SELECT * FROM sqlite_master WHERE name = '{tbl_name}'")
            ret = res.fetchone() is not None
        return ret
    
    def _on_init_check_meta_table(self):
        """Check if the basic meta table is correctly setup in DB"""

        if not self._exist_table(TableName.Meta.run_log):
            with self.conn as conn:
                conn.execute(f"""
                    CREATE TABLE "{TableName.Meta.run_log}" (
                            "ticker_name"	TEXT,
                            "ticker_type"	TEXT,
                            "run_date"	DATE,
                            "run_datetime"	TIMESTAMP,
                            "run_intraday_version"	INTEGER,
                            "run_status"	INTEGER,
                            "task_name"	TEXT
                    );""")

            

    # def _drop_all_tables(self):
