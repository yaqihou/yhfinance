
import os
import sqlite3

import pandas as pd

DB_NAME = os.path.join(
    os.environ.get('HOME', '.'), 'Dropbox', '66-DBs', 'FinDB.db'
)

class DBMessenger:

    _conn_dict: dict[str, sqlite3.Connection] = {}

    def __init__(self, db_name: str = DB_NAME):
        self._db_name = db_name

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
    
    # TODO - def check_db_on_init
