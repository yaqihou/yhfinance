from enum import Enum
import datetime as dt
from dataclasses import dataclass
import dataclasses


from typing import Optional


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


class TickerGroup(Enum):
    """Not a stirct allocation as the traditional sector definition, more like a moniker"""

    FINANCIAL = 'Financial'
    TECH = 'Tech'
    SEMI_CONDUCTOR = 'Semiconductor'
    ENERGY = 'Energy'
    CONSUMER = 'Consumer'
    HEALTH = 'HealthCare'
    INDUSTRIAL = 'INDUSTRIAL'
    GAME = 'Game'

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

    @property
    def is_intraday(self):
        return self.value[-1] in {'m', 'h'}

    @classmethod
    def get_all_intraday_intervals(cls):
        return [cls.MIN_1, cls.MIN_2, cls.MIN_5,
                cls.MIN_15, cls.MIN_30, cls.MIN_60, cls.MIN_90]
        


class DownloadSwitch:

    HISTORY: int = 1
    INFO: int = 1 << 1
    NEWS: int = 1 << 2
    HOLDER: int = 1 << 3
    FINANCIAL: int = 1 << 4
    RATING: int = 1 << 5
    OPTION: int = 1 << 6

    ALL: int = HISTORY | INFO | NEWS | HOLDER | FINANCIAL | RATING | OPTION


@dataclass
class HistoryExtraOptions:
    # https://github.com/ranaroussi/yfinance/wiki/Ticker

    # Include Pre and Post market data in results?
    prepost: bool = False
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

