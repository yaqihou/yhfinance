
import os
import sqlite3

import logging
from typing import Callable

import pandas as pd
import datetime as dt

from .defs import TableName, MetaTableDefinition

DB_NAME = os.path.join(
    os.environ.get('HOME', '.'), 'Dropbox', '66-DBs', 'FinDB.db'
)

logger = logging.getLogger("yfinance-backup.db")


class DB:
    
    _conn_dict: dict[str, sqlite3.Connection] = {}

    def __init__(self, db_name: str = DB_NAME):
        self._db_name = db_name

        self._on_init_check_meta_table()

    @property
    def conn(self) -> sqlite3.Connection:
        if self._db_name not in self._conn_dict:
            self._conn_dict[self._db_name] = sqlite3.connect(self._db_name)
        return self._conn_dict[self._db_name]

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
                logger.debug('MetaTable %s does not exist, creating a new one', TableName.Meta.run_log)
                conn.execute(MetaTableDefinition.run_log)

class DBMessenger:

    def __init__(self, db_name: str = DB_NAME):
        self._db_name = db_name
        self.db = DB(db_name)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        return

    @property
    def conn(self):
        return self.db.conn
    
    def read_sql(self, sql: str) -> pd.DataFrame:
        # TODO - add exception treatment
        # TODO - need to check if the columns are consistent
        return pd.read_sql(sql, self.conn)

            
class DBMaintainer:

    def __init__(self, db_name: str = DB_NAME):
        self._db_name = db_name
        self.db = DB(db_name)

    def maintain_tickers_meta(self):
        
        # Step 1 - make sure all tickers in DB are covered in meta_tickers
        # The only source of new unknown tickers are in 
        
        # Step 2 - fetch all information about each ticker

        # Step 3 - 

        return

    # TODO - def maintain_tasks_meta

    def _for_each_table(self, func: Callable, include_meta: bool = False):

        assert callable(func)

        ret = {}
        for tbl_name in TableName.to_list(include_meta=include_meta):
            if self.db._exist_table(tbl_name):
                ret[tbl_name] = func(tbl_name)
        return ret

    def _report_status(self, tbl_name):
        df = self.db.read_sql(f"SELECT * FROM {tbl_name}")
        
        # Add other information we may be interested
        return df.shape, df.columns 

    def report_status(self):

        tbls_status = self._for_each_table(self._report_status)
        # TODO add a summary printout
        return
        
    def _maintain_unique_entries(self, tbl_name: str, dryrun: bool = False):
        
        logger.info('Cleaning up duplicate rows in table %s', tbl_name)

        df = self.db.read_sql(f"SELECT * FROM {tbl_name}")
        _old_shape = df.shape
        df = df.drop_duplicates(ignore_index=True)
        _new_shape = df.shape

        if _old_shape[0] == _new_shape[0]:
            logger.info('No cleanup neede - all entries are unique for table %s',
                        _old_shape[0], _new_shape[0], tbl_name)
        else:
            logger.info('Cleaned up duplicated rows %d -> %d in table %s',
                        _old_shape[0], _new_shape[0], tbl_name)

        if not dryrun:
            df.to_sql(tbl_name, self.db.conn, if_exists='replace', index=False)

        return

    def maintain_unique_entries(self, dryrun=False):
        self._for_each_table(
            lambda tbl_name: self._maintain_unique_entries(tbl_name, dryrun=dryrun)
        )

    def _drop_all_tables(self, include_meta: bool = False):

        with self.db.conn as conn:
            for tbl_name in TableName.to_list(include_meta=include_meta):
                logger.debug('Try to drop table %s', tbl_name)
                conn.execute(f'DROP IF EXISTS {tbl_name}')
