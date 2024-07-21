
import abc
from typing import Literal, Optional, Union, get_args
from collections.abc import Iterable

import pandas as pd
import datetime as dt

from pandas._libs.interval import intervals_to_interval_bounds
from pandas.io.formats.format import _IntArrayFormatter

from yhfinance.const.db.table_name import TableName
from yhfinance.const.db.col_name import MetaColName
from yhfinance.utils import parse_input_datetime
from yhfinance.const.tickers import Interval
from yhfinance.db_utils import DBConfig, DBFetcher

TimeInput = Union[int , str , dt.datetime , dt.date , pd.Timestamp]

class BaseLoader(abc.ABC):

    _fetcher_regs: dict[str, DBFetcher] = {}
    
    def __init__(self,
                 ticker_name: str,
                 db_name: str = DBConfig.DB_NAME,
                 ):
        self.ticker_name: str = ticker_name
        self._db_name = db_name

        if BaseLoader._fetcher_regs.get(db_name) is None:
            BaseLoader._fetcher_regs[db_name] = DBFetcher(db_name)

    @property
    def fetcher(self) -> DBFetcher:
        if BaseLoader._fetcher_regs.get(self._db_name) is None:
            BaseLoader._fetcher_regs[self._db_name] =DBFetcher(self._db_name)
            
        return BaseLoader._fetcher_regs[self._db_name]

    @property
    @abc.abstractmethod
    def _tbl_name(self) -> str:
        """Return the table name for this loader"""

    def _build_where_clause(self) -> str:
        _where_conds = [
            self._build_ticker_name_cond(),
            self._build_date_range_cond(MetaColName.RUN_DATETIME, self._run_start, self._run_end),
            self._build_extra_conds()
        ]

        _where_clause = '\n    AND '.join(filter(lambda x: len(x) > 0, _where_conds))
        if _where_conds:
            return f"WHERE {_where_clause}"
        else:
            return ""

    @abc.abstractmethod
    def _build_extra_conds(self) -> str:
        """Override to support extra conditions pass to WHERE
        return "" if nothing extra needed
        """

    def _build_ticker_name_cond(self) -> str:
        ret = f"{MetaColName.TICKER_NAME} = '{self.ticker_name}'"
        if self._ignore_ticker_name_case:
            ret += "  COLLATE NOCASE"
        return ret

    def load(self, *,
             run_start: Optional[TimeInput] = None,
             run_end: Optional[TimeInput] = None,
             ignore_ticker_name_case: bool = True,
             ignore_meta_column: bool | list[str] = [
                 MetaColName.RUN_DATE, MetaColName.RUN_DATETIME,
                 MetaColName.TASK_NAME]
             ) -> pd.DataFrame:
        """Load the query built with given input, in derived class, implment the initiate
        so that the qurey building function have all variables set up

        ignore_meta_column: True to ignore all defined MetaCols or a list of column to ignore
        """

        self._ignore_ticker_name_case = ignore_ticker_name_case
        self._ignore_meta_column = ignore_meta_column
        self._run_start = run_start
        self._run_end = run_end

        sql = self._build_query()
        df = self.fetcher.read_sql(sql)
        if ignore_meta_column:
            df = self._filter_meta_column(df, ignore_meta_column)

        # Apply datatype
        df['Datetime'] = df['Datetime'].apply(lambda x: dt.datetime.strptime(x, '%Y-%m-%d %H:%M:%S%z'))
        df['Date'] = df['Date'].apply(lambda x: dt.datetime.strptime(x, '%Y-%m-%d'))
            
        return df
        
    def _build_query(self):

        _query_list = [
            # TODO - add customizeable columns
            'SELECT *',
            f'FROM {self._tbl_name}',
            self._build_where_clause()
        ]
        return '\n'.join(filter(lambda x: len(x) > 0, _query_list))

    @staticmethod
    def _filter_meta_column(
            df: pd.DataFrame,
             ignore_meta_column: bool | list[str] = True
    ) -> pd.DataFrame:
        """Return the dataframe without job meta column"""

        excluded_cols: set[str]
        if isinstance(ignore_meta_column, bool):
            excluded_cols = set(MetaColName.to_list())
        elif isinstance(ignore_meta_column, (list, tuple)):
            excluded_cols = set(map(str, ignore_meta_column))
        else:
            raise ValueError("Input ignore_meta_column is invalid")
            
        new_cols: list[str] = list(
            filter(
                lambda x: x not in excluded_cols,
                df.columns
            ))
        return df[new_cols]

    @staticmethod
    def _build_date_range_cond(
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

        if _range_conds:
            return "(" + " AND ".join(_range_conds) + ")"
        else:
            return ""

    @staticmethod
    def _build_date_points_cond(
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


class _OptionLoader(BaseLoader):

    def load(self, *,
             expires: Optional[list[TimeInput] | TimeInput] = None,
             expire_start: Optional[TimeInput] = None,
             expire_end: Optional[TimeInput] = None,
             run_start: Optional[TimeInput] = None,
             run_end: Optional[TimeInput] = None,
             ignore_ticker_name_case: bool = True,
             ignore_meta_column: bool | list[str] = True
             ) -> pd.DataFrame:

        self._expires: Optional[list[TimeInput] | TimeInput] = expires
        self._expire_start: Optional[TimeInput] = expire_start
        self._expire_end: Optional[TimeInput] = expire_end

        return super().load(run_start=run_start,
                            run_end=run_end,
                            ignore_ticker_name_case=ignore_ticker_name_case,
                            ignore_meta_column=ignore_meta_column)

    def _build_extra_conds(self) -> str:

        _expire_conds = []

        _expires_str = self._build_date_points_cond('expire', self._expires, is_date=True)
        if _expires_str:  _expire_conds.append(_expires_str)

        _expire_rng_str = self._build_date_range_cond(
            'expire', self._expire_start, self._expire_end, is_date=True)
        if _expire_rng_str:  _expire_conds.append(_expire_rng_str)

        if _expire_conds:
            return "(" + " OR ".join(_expire_conds) + ")"
        else:
            return ""


class CallOptionLoader(_OptionLoader):

    @property
    def _tbl_name(self) -> str:
        return TableName.Option.CALLS


class PutOptionLoader(_OptionLoader):

    @property
    def _tbl_name(self) -> str:
        return TableName.Option.PUTS


class _HistoryLoader(BaseLoader):

    def __init__(self,
                 ticker_name: str,
                 interval: Interval | str,
                 db_name: str = DBConfig.DB_NAME):
        super().__init__(ticker_name, db_name)

        if isinstance(interval, str):
            self.interval = Interval(interval)
        elif isinstance(interval, Interval):
            self.interval = interval
        else:
            raise ValueError('Input interval should be a str or Interval Enum')

    @property
    def _tbl_name(self):
        return TableName.History.PRICE_TABLE_MAPPING[self.interval.value]


_T_PERIOD_TYPE = Literal["pre", "post", "regular", "all"]
class IntradayHistoryLoader(_HistoryLoader):

    def load(self, *,
             start: Optional[TimeInput] = None,
             end: Optional[TimeInput] = None,
             period_type: Optional[_T_PERIOD_TYPE] = None,
             run_start: Optional[TimeInput] = None,
             run_end: Optional[TimeInput] = None,
             ignore_ticker_name_case: bool = True,
             ignore_meta_column: bool | list[str] = True
             ) -> pd.DataFrame:

        self._start: Optional[TimeInput] = start
        self._end: Optional[TimeInput] = end

        if period_type and period_type not in get_args(_T_PERIOD_TYPE):
            raise ValueError(f"Input period_type should be one of {', '.join(get_args(_T_PERIOD_TYPE))}."
                             f" Current value: {period_type}")
        
        self._period_type: str = period_type or "all"

        return super().load(run_start=run_start,
                            run_end=run_end,
                            ignore_ticker_name_case=ignore_ticker_name_case,
                            ignore_meta_column=ignore_meta_column)
            

    def _build_extra_conds(self) -> str:
        conds = []

        _date_range = self._build_date_range_cond('Datetime', self._start, self._end, is_date=False)
        if _date_range:  conds.append(_date_range)

        if self._period_type != "all":
            conds.append(f"period_type = '{self._period_type}'")

        if conds:
            return "\n    AND".join(conds)
        else:
            return ""


class DayHistoryLoader(_HistoryLoader):

    def load(self, *,
             start: Optional[TimeInput] = None,
             end: Optional[TimeInput] = None,
             run_start: Optional[TimeInput] = None,
             run_end: Optional[TimeInput] = None,
             ignore_ticker_name_case: bool = True,
             ignore_meta_column: bool | list[str] = True
             ) -> pd.DataFrame:

        self._start: Optional[TimeInput] = start
        self._end: Optional[TimeInput] = end

        return super().load(run_start=run_start,
                            run_end=run_end,
                            ignore_ticker_name_case=ignore_ticker_name_case,
                            ignore_meta_column=ignore_meta_column)
            

    def _build_extra_conds(self) -> str:
        return self._build_date_range_cond('Date', self._start, self._end, is_date=True)

class DBLoader:

    def __init__(self, ticker_name: str, db_name: str = DBConfig.DB_NAME):
        self.ticker_name = ticker_name
        self._db_name = db_name

        self._call_option_loader: Optional[CallOptionLoader] = None
        self._put_option_loader: Optional[CallOptionLoader] = None
        # self._option_expirations_loader: Optional[CallOptionLoader] = None

        self._hist_loaders: dict[str, IntradayHistoryLoader | DayHistoryLoader] = {}

    def get_options(
            self,
            is_call: bool = True,
            expires: Optional[list[TimeInput] | TimeInput] = None,
            expire_start: Optional[TimeInput] = None,
            expire_end: Optional[TimeInput] = None,
            run_start: Optional[TimeInput] = None,
            run_end: Optional[TimeInput] = None,
            ignore_meta_column: bool | list[str] = True
    ) -> pd.DataFrame:

        _LoaderCls = CallOptionLoader if is_call else PutOptionLoader
        loader = self._call_option_loader if is_call else self._put_option_loader

        if loader is None:
            loader = _LoaderCls(self.ticker_name, self._db_name)

        return loader.load(
            expires=expires, expire_start=expire_start, expire_end=expire_end,
            run_start=run_start, run_end=run_end,
            ignore_meta_column=ignore_meta_column,
        )

    def get_history(
            self,
            interval: Interval | str = Interval.MINUTE,
            start: Optional[TimeInput] = None,
            end: Optional[TimeInput] = None,
            period_type: Optional[_T_PERIOD_TYPE] = None,
            run_start: Optional[TimeInput] = None,
            run_end: Optional[TimeInput] = None,
            ignore_ticker_name_case: bool = True,
            ignore_meta_column: bool | list[str] = True
    ):

        _interval: Interval
        if isinstance(interval, str):
            _interval = Interval(interval)
        elif isinstance(interval, Interval):
            _interval = interval
        else:
            raise ValueError('Input interval should be a str or Interval Enum')

        _LoaderCls = IntradayHistoryLoader if _interval.is_intraday else DayHistoryLoader

        if self._hist_loaders.get(_interval.value) is None:
            self._hist_loaders[_interval.value] = _LoaderCls(
                self.ticker_name, _interval, db_name=self._db_name)

        args = {
            'start'                   : start,
            'end'                     : end,
            'run_start'               : run_start,
            'run_end'                 : run_end,
            'ignore_ticker_name_case' : ignore_ticker_name_case,
            'ignore_meta_column'      : ignore_meta_column
        }

        if _interval.is_intraday:
            args['period_type'] = period_type
        else:
            if period_type is not None:
                print('Warning: period_type argument is ignored for day history')
                

        return self._hist_loaders[_interval.value].load(**args)
        
