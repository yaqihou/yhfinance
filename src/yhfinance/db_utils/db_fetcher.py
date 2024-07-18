import textwrap
from typing import Optional, Union
from collections.abc import Iterable
import datetime as dt
import pandas as pd

from yhfinance.const.db import TableName
from yhfinance.const.db.col_name import MetaColName
from yhfinance.const.tickers import Interval
from yhfinance.logger import MyLogger
from yhfinance.utils import parse_input_datetime

from . import DB, DBConfig

# TODO - move to a centralized place
TimeInput = Union[int , str , dt.datetime , dt.date , pd.Timestamp]
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
        logger.debug("Fecthing df using the following query:\n%s", sql)
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

    def _filter_meta_column(self, df: pd.DataFrame) -> pd.DataFrame:
        """Return the dataframe without job meta column"""
        return df

    @staticmethod
    def _build_where_date_range_clause(
            col_name: str,
            start: Optional[TimeInput] = None,
            end: Optional[TimeInput] = None,
            is_date: bool = False
    ) -> str:
        """Return the where clause to limit search scope >= start AND <= end
        """
        
        _func = 'date' if is_date else 'datetime'
        _start = None if start is None else parse_input_datetime(start)
        _end = None if end is None else parse_input_datetime(end)

        _range_conds = []
        if _start is not None:
            _range_conds.append(f"{_func}({col_name}) >= {_func}('{_start}')")

        if _end is not None:
            _range_conds.append(f"{_func}({col_name}) <= {_func}('{_end}')")

        return "(" + " AND ".join(_range_conds) + ")"

    @staticmethod
    def _build_where_date_points_clause(
            col_name: str,
            dates: Optional[TimeInput | list[TimeInput]] = None,
            is_date: bool = False
    ) -> str:
        """Return the where clause to limit search scope IN (<given list / single date>)
        """

        _func = 'date' if is_date else 'datetime'

        if dates is None:
            _dates = None
        else:
            if isinstance(dates, (list, tuple)):
                _dates = list(map(parse_input_datetime, dates))
            else:
                _dates = [parse_input_datetime(dates)]

        if _dates is not None:
            _dates_str = ', '.join(map(lambda x: f"{_func}('{x}')", _dates))
            return (f"{_func}({col_name}) IN ({_dates_str})")
        else:
            return ""


    def get_options(
            self,
            ticker_name: str,
            is_call: bool = True,
            expires: Optional[list[TimeInput] | TimeInput] = None,
            expire_start: Optional[TimeInput] = None,
            expire_end: Optional[TimeInput] = None,
            run_start: Optional[TimeInput] = None,
            run_end: Optional[TimeInput] = None,
            ignore_meta_column: bool = True
    ) -> pd.DataFrame:

        _tbl_name = TableName.Option.CALLS if is_call else TableName.Option.PUTS

        _expire_conds = []

        _expires_str = self._build_where_date_points_clause('expire', expires, is_date=True)
        if _expires_str:  _expire_conds.append(_expires_str)

        _expire_rng_str = self._build_where_date_range_clause('expire', expire_start, expire_end, is_date=True)
        if _expire_rng_str:  _expire_conds.append(_expire_rng_str)

        _where_conds = [
            f"{MetaColName.TICKER_NAME} = '{ticker_name}' COLLATE NOCASE",
            self._build_where_date_range_clause(MetaColName.RUN_DATETIME, run_start, run_end),
            "(" + " OR ".join(_expire_conds) + ")"
        ]

        _where_clause = '\n    AND '.join(filter(lambda x: len(x) > 0, _where_conds))
        sql = textwrap.dedent(
            f"""SELECT *
            FROM {_tbl_name}
            WHERE {_where_clause}
            ;""")

        print(sql)
        return self.fetcher.read_sql(sql)

    def get_intraday_history(self):
        pass

    def get_day_histroy(self):
        pass


    def _clean_up_history_duplicates(self):
        pass
