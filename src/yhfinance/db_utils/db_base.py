
import sqlite3

from typing import Literal 

import pandas as pd

from yhfinance.const.databackup import JobSetup
from yhfinance.const.db import TableName, MetaTableDefinition
from yhfinance.logger import MyLogger

from . import DBConfig

logger = MyLogger.getLogger("db_utils")


class DB:

    _conn_dict: dict[str, sqlite3.Connection] = {}

    # TODO - make this one a module level configurable variable
    def __init__(self, db_name: str = DBConfig.DB_NAME):
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

    def add_job_status(self, job: JobSetup, status: int):
        _df = pd.DataFrame.from_dict({
            'run_status': [status],
            **{k: [v] for k, v in job.metainfo.items()}
        })

        with self.conn as conn:
            _df.to_sql(TableName.Meta.run_log, conn, if_exists='append', index=False)

    def add_df(self, df: pd.DataFrame, table_name: str, if_exists: Literal['append', 'fail', 'replace'] = 'append'):

        logger.debug('Dumping DataFrame to %s', table_name)

        if if_exists == 'append':

            if self._exist_table(table_name):
               self._reconcile_df_column_names(df, table_name)

            try:
                with self.conn as conn:
                    df.to_sql(table_name, conn, if_exists=if_exists, index=False)
            except Exception as e:
                logger.error('Encountered error when dumping DataFrame to %s', table_name, exc_info=e)
            else:
                logger.info('Successfully dump DataFrame into %s', table_name)
        else:
            with self.conn as conn:
                df.to_sql(table_name, conn, if_exists=if_exists, index=False)


    def _reconcile_df_column_names(self, df: pd.DataFrame, table_name: str):

        with self.conn as conn:
            df_table = pd.read_sql(
                 f'PRAGMA table_info({table_name});',
                 conn, index_col='cid')

        current_cols = set(df_table['name'].to_list())

        new_cols = []
        for col in df.columns:
            if col not in current_cols:
                _type_name = df[col].dtype.name
                if 'float' in _type_name:
                    _type = 'REAL'
                elif 'int' in _type_name:
                    _type = 'INTEGER'
                elif 'datetime' in _type_name:
                    _type = 'TIMESTAMP'
                else:
                    _type = 'TEXT'
                new_cols.append((col, _type))
            
        with self.conn as conn:
            for col, _type in new_cols:
                cmd = f'ALTER TABLE {table_name} ADD COLUMN {col} {_type};'
                logger.info(
                    f'DataFrame need new column %s (%s), adding it using the query below\n%s',
                    col, _type, cmd)
                conn.execute(cmd)

        return
