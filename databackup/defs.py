from dataclasses import dataclass, Field
import dataclasses
import datetime as dt

from enum import Enum

from typing import Optional


class BackupFrequency(Enum):

    # Run anyway
    ONE_TIME = dt.timedelta(seconds=0)
    ONCE = dt.timedelta(seconds=0)
    AD_HOC = dt.timedelta(seconds=0)

    HOURLY = dt.timedelta(hours=1)
    HOUR_1 = dt.timedelta(hours=1)

    HOUR_2 = dt.timedelta(hours=2)
    HOUR_4 = dt.timedelta(hours=4)

    QUARTER_DAY = dt.timedelta(hours=6)
    HOUR_6 = dt.timedelta(hours=6)

    ONE_THIRD_DAY = dt.timedelta(hours=8)
    HOUR_8 = dt.timedelta(hours=8)

    HALF_DAY = dt.timedelta(hours=12)
    HOUR_12 = dt.timedelta(hours=12)

    DAILY = dt.timedelta(days=1)
    WEEKLY = dt.timedelta(days=7)
    MONTHLY = dt.timedelta(days=30)
    QUARTERLY = dt.timedelta(days=120)
    SEMIANNUAL = dt.timedelta(days=180)
    ANNUAL = dt.timedelta(days=360)
    

@dataclass
class HistoryExtraOptions:
    # https://github.com/ranaroussi/yfinance/wiki/Ticker

    # Include Pre and Post market data in results?
    prepost: bool = True
    # Include Dividends and Stock Splits in results?
    actions: bool = False	
    # Dividend-adjust all OHLC automatically?	
    auto_adjust: bool = True	
    # Detect problems in price data and repair. See Wiki page for details
    repair: bool = True	
    # Keep NaN rows returned by Yahoo?
    keepna: bool = False	
    # Proxy server URL scheme.	
    proxy: Optional[str] = None	
    # Round values using Yahoo-suggested precision?	
    rounding: bool = False	
    # Stop waiting for response after N seconds.	
    timeout: Optional[float] = 10	
    # Raise errors as exceptions instead of printing?
    raise_errors: bool = False

    def to_dict(self):
        return {field.name: getattr(self, field.name) for field in dataclasses.fields(self)}


class TickerType(Enum):

    STOCK = 'Stock'
    ETF = 'ETF'
    Index = 'Index'
    Crypto = 'Crypto'


class Period(Enum):

    D01 = '1d'
    D05 = '5d'

    M01 = '1mo'
    M03 = '3mo'
    M06 = '6mo'

    Y01 = '1y'
    Y02 = '2y'
    Y05 = '5y'
    Y10 = '10y'

    YTD = 'ytd'
    MAX = 'max'


class Interval(Enum):

    MINUTE = "1m"
    MIN_1 = '1m'
    MIN_2 = '2m'
    MIN_5 = '5m'

    MIN_15 = '15m'
    MIN_30 = '30m'
    MIN_60 = '60m'
    MIN_90 = '90m'

    HOUR = '1h'
    HR_1 = '1h'

    DAY = '1d'
    DAY_1 = '1d'
    DAY_5 = '5d'
    DAY_7 = '1wk'

    WEEK = '1wk'
    WK_1 = '1wk'

    MONTH = '1mo'
    MON_1 = '1mo'
    MON_3 = '3mo'


class DownloadSwitch:

    HISTORY: int = 1
    INFO: int = 1 << 1
    NEWS: int = 1 << 2
    HOLDER: int = 1 << 3
    FINANCIAL: int = 1 << 4
    RECOMMENDATION: int = 1 << 5
    OPTION: int = 1 << 6

    PRICE: int = HISTORY | INFO | HOLDER
    ALL: int = HISTORY | INFO | NEWS | HOLDER | FINANCIAL | RECOMMENDATION | OPTION


@dataclass(kw_only=True)
class BaseTask:

    # Task Info
    backup_freq: BackupFrequency
    download_switch: int

    name: Optional[str] = None
    # History Input
    interval: Interval = Interval.DAY
    past_days: int = -1
    # The below arguments would be useful to create ad hoc tasks (like to run for a customized period)
    period: Optional[Period] = None  # Period have the highest priority
    start: Optional[dt.datetime | str | int] = None
    end: Optional[dt.datetime | str | int] = None
    history_extra_options: HistoryExtraOptions = HistoryExtraOptions()
    
    # Download Price Data by default
    download_full_text_news: bool = False

    def __post_init__(self):
        if self.name is None:
            self.name = f'adhoc_task_{dt.datetime.today()}'

    def get_args(self):
        return {
            'download_switch': self.download_switch,
            'download_full_text_news': self.download_full_text_news
        }


@dataclass(kw_only=True)
class HistoryTask(BaseTask):
    download_switch: int = DownloadSwitch.PRICE

    def get_args(self):
        return {
            **super().get_args(),
            'interval': self.interval,
            'history_extra_options': self.history_extra_options
        }


@dataclass(kw_only=True)
class IntraDayHistoryTask(HistoryTask):
    past_days: int = 0 # Default to backup current days' data
    backup_freq: BackupFrequency = BackupFrequency.DAILY
    download_switch: int = DownloadSwitch.HISTORY


@dataclass
class TickerConfig:
    ticker_name: str
    ticker_type: TickerType
    added_date: dt.date| str

    tasks: list[BaseTask]
    notes: str = ""


class JobStatus:

    INIT = 0
    SUCCESS = 1
    FAIL = 2


@dataclass(kw_only=True)
class JobSetup:

    # Basic setup - related with Ticker 
    ticker_name: str
    ticker_type: TickerType

    # Run setup - extra information
    run_datetime: dt.datetime 
    run_intraday_version: int
    task: BaseTask | HistoryTask | IntraDayHistoryTask

    # Setup for history
    # NOTE - similar info in task field as well,
    #        but below are parsed from task to be used directly
    interval: Interval = Interval.DAY
    period: Optional[Period] = None
    start: Optional[dt.date | str | int] = None
    end: Optional[dt.date | str | int] = None
    history_extra_options: HistoryExtraOptions = HistoryExtraOptions()

    # Download setup
    download_full_text_news: bool = False
    download_switch: int = DownloadSwitch.ALL

    @property
    def run_date(self) -> dt.date:
        return self.run_datetime.date()

    # @property
    # def is_intraday(self) -> bool:

    def __post_init__(self):
        # Only need to check if download history data
        if self.download_switch & DownloadSwitch.HISTORY:
            if self.period is None:
                assert self.end is not None and self.start is not None
            else:
                assert self.end is None and self.start is None
    
    def get_history_args(self) -> dict:

        args = {
            'interval': self.interval.value,
            **self.history_extra_options.to_dict()
        }

        if self.period is not None:
            args['period'] = self.period.value
        else:
            args['period'] = None
            args['start'] = self.start
            args['end'] = self.end

        return args

    @property
    def history_args(self) -> dict:
        return self.get_history_args()

    @property
    def history_range(self) -> str:
        if self.period is None:
            return f"{self.start} to {self.end}"
        else:
            return f"period of {self.period.value}"

    @property
    def metainfo(self) -> dict:
        # TODO - when dumping to DB, using ticker name is not good. Need to replace it
        #        with index
        return {
            'ticker_name': self.ticker_name,
            'ticker_type': self.ticker_type.value,
            'run_date': self.run_date,
            'run_datetime': self.run_datetime,
            'run_intraday_version': self.run_intraday_version,
            'task_name': self.task.name
        }

        
        
