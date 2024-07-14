
from enum import Enum
import datetime as dt
from dataclasses import dataclass, field

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


@dataclass
class BackupCondition:

    on_weekday: set[int] = field(default_factory=lambda: {0, 1, 2, 3, 4, 5, 6})
    on_time_range: list[tuple[Optional[dt.time], Optional[dt.time]]] = field(default_factory=lambda: [(dt.time.min, dt.time.max)])

    def check(self, run_datetime: dt.datetime) -> dict[str, bool]:
        """Return if the given datetime satisfy the condition"""

        satisfy_on_weekday: bool = run_datetime.weekday() in self.on_weekday

        satisfy_on_time_range: bool = False
        _rundate = run_datetime.date()
        for _s, _e in self.on_time_range:
            if _s is None:  _s = dt.time.min
            if _e is None:  _e = dt.time.max

            if (
                    (run_datetime >= dt.datetime.combine(_rundate, _s))
                    and (run_datetime < dt.datetime.combine(_rundate, _e))
            ):
                satisfy_on_time_range = True
                break

        return {
            'on_weekday': satisfy_on_weekday,
            'on_time_range': satisfy_on_time_range
        }

# Below defined some common conditions
_weekday = {0, 1, 2, 3, 4}

_time_range_after_market_close_normal: list[tuple[Optional[dt.time], Optional[dt.time]]] = [
    (dt.time(hour=16), dt.time.max)]
_time_range_after_market_close_extended: list[tuple[Optional[dt.time], Optional[dt.time]]] = [
    (dt.time(hour=20), dt.time.max)]

_time_range_before_market_open_normal: list[tuple[Optional[dt.time], Optional[dt.time]]] = [
    (dt.time.min, dt.time(hour=9, minute=30))]
_time_range_before_market_open_extended: list[tuple[Optional[dt.time], Optional[dt.time]]] = [
    (dt.time.min, dt.time(hour=4))]

_time_range_when_market_close_normal: list[tuple[Optional[dt.time], Optional[dt.time]]] = [
    (dt.time.min, dt.time(hour=9, minute=30)),
    (dt.time(hour=16), dt.time.max)]
_time_range_when_market_close_extended: list[tuple[Optional[dt.time], Optional[dt.time]]] = [
    (dt.time.min, dt.time(hour=4)),
    (dt.time(hour=20), dt.time.max)]

bc_all = BackupCondition()
bc_weekday = BackupCondition(on_weekday=_weekday)
bc_weekday_when_market_close_normal = BackupCondition(
    on_weekday=_weekday, on_time_range=_time_range_when_market_close_normal)
bc_weekday_when_market_close_extended = BackupCondition(
    on_weekday=_weekday, on_time_range=_time_range_when_market_close_extended)

bc_weekday_after_market_close_normal = BackupCondition(
    on_weekday=_weekday, on_time_range=_time_range_after_market_close_normal)
bc_weekday_after_market_close_extended = BackupCondition(
    on_weekday=_weekday, on_time_range=_time_range_after_market_close_extended)


bc_friday_after_market_close_normal = BackupCondition(
    on_weekday={4}, on_time_range=_time_range_after_market_close_normal)
bc_friday_after_market_close_extended = BackupCondition(
    on_weekday={4}, on_time_range=_time_range_after_market_close_extended)

bc_weekday_before_market_open_normal = BackupCondition(
    on_weekday=_weekday, on_time_range=_time_range_before_market_open_normal)
bc_weekday_before_market_open_extended = BackupCondition(
    on_weekday=_weekday, on_time_range=_time_range_before_market_open_extended)


@dataclass(kw_only=True)
class BaseTask:

    # Task Info
    backup_freq: BackupFrequency
    backup_cond: BackupCondition
    
    download_switch: int

    name: Optional[str] = None
    # History Input
    interval: Interval = Interval.DAY
    past_days: int = -1
    end_day_offset: int = 0  # if 0, end day will be EoD of run date, which may not be good for Crypto
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

    download_switch: int = DownloadSwitch.HISTORY

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
    backup_cond: BackupCondition = bc_weekday_after_market_close_extended

    history_extra_options: HistoryExtraOptions = HistoryExtraOptions(prepost=True)


@dataclass(kw_only=True)
class IntraDayHistoryTaskCrypto(HistoryTask):
    """Since Crypto is trading 24/7, we need to define a special rule"""
    
    past_days: int = 0 # Default to backup current days' data
    end_day_offset: int = 1 # Today's data will always be incomplete
    backup_freq: BackupFrequency = BackupFrequency.DAILY
    backup_cond: BackupCondition = bc_all

    history_extra_options: HistoryExtraOptions = HistoryExtraOptions(prepost=True)
