
from enum import Enum
import datetime as dt
from dataclasses import dataclass

from typing import Optional

from .tickers import DownloadSwitch, Period, Interval, HistoryExtraOptions


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
