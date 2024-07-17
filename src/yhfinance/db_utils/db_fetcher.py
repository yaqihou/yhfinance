from typing import Optional
import datetime as dt
import pandas as pd

from yhfinance.const.db import TableName
from yhfinance.const.tickers import Interval
from yhfinance.logger import MyLogger

from . import DB, DBConfig

logger = MyLogger(DBConfig.LOGGER_NAME)

class DBFetcher:

    def __init__(self, db_name: str = DBConfig.DB_NAME):
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

    # --------------------------------
    # Create a bunch of shortcuts to load data
    # TODO - considering if we need to factor them into a separate class

class DBHistoryFetcher:

    def __init__(self, db_name: str = DBConfig.DB_NAME):
        self._db_name = db_name
        self.fetcher = DBFetcher(db_name)

    # def get_history(
    #         self,
    #         ticker_name: str,
    #         interval: str | Interval,
    #         start: Optional[str | dt.datetime | int] = None,
    #         end: Optional[str | dt.datetime | int] = None
    # ):
    #     df = self.fetcher.read_sql(f"""
    #     SELECT
    #     """)
    #     pass

    def get_options(
            self,
            ticker_name: str,
            start: Optional[str | dt.datetime | int],
            end: Optional[str | dt.datetime | int]
    ):
        pass

    def get_intraday_history(self):
        pass

    def _clean_up_history_duplicates(self):
        pass
