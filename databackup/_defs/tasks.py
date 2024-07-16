
from enum import Enum
import datetime as dt
from dataclasses import dataclass, field

from typing import Optional

from databackup.logger import MyLogger

from .tickers import DownloadSwitch, Period, Interval, HistoryExtraOptions

logger = MyLogger.getLogger('utils')


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

    def check(self,
              run_datetime: dt.datetime,
              buffer_time: dt.timedelta = dt.timedelta(minutes=30)) -> dict[str, bool]:
        """Return if the given datetime satisfy the condition, buffer_time is applied
        both ways (earlier start time and later end time) to time range with cap to current day

        """

        satisfy_on_weekday: bool = run_datetime.weekday() in self.on_weekday

        satisfy_on_time_range: bool = False
        _rundate = run_datetime.date()
        for _s, _e in self.on_time_range:
            if _s is None:  _s = dt.time.min
            if _e is None:  _e = dt.time.max

            _open = max(
                dt.datetime.combine(_rundate, dt.time.min),
                dt.datetime.combine(_rundate, _s) - buffer_time
            )

            _close = min(
                dt.datetime.combine(_rundate, dt.time.max),
                dt.datetime.combine(_rundate, _e) + buffer_time
            )

            if ((run_datetime >= _open) and (run_datetime < _close)):
                satisfy_on_time_range = True

            logger.debug('Comparing %s with boundary [%s, %s): %s',
                         str(run_datetime), str(_open), str(_close), str(satisfy_on_time_range))
            if satisfy_on_time_range:
                break
                


        return {
            'on_weekday': satisfy_on_weekday,
            'on_time_range': satisfy_on_time_range
        }

# Below defined some common conditions
_weekday = {0, 1, 2, 3, 4}

_time_range_after_market_close_normal: list[tuple[Optional[dt.time], Optional[dt.time]]] = [
    (dt.time(hour=16), dt.time.max)]
_time_range_after_market_close_extend: list[tuple[Optional[dt.time], Optional[dt.time]]] = [
    (dt.time(hour=20), dt.time.max)]

_time_range_during_market_open_normal: list[tuple[Optional[dt.time], Optional[dt.time]]] = [
    (dt.time(hour=9, minute=30), dt.time(hour=16))]
_time_range_during_market_open_extend: list[tuple[Optional[dt.time], Optional[dt.time]]] = [
    (dt.time(hour=4), dt.time(hour=20))]

_time_range_before_market_open_normal: list[tuple[Optional[dt.time], Optional[dt.time]]] = [
    (dt.time.min, dt.time(hour=9, minute=30))]
_time_range_before_market_open_extend: list[tuple[Optional[dt.time], Optional[dt.time]]] = [
    (dt.time.min, dt.time(hour=4))]

_time_range_when_market_close_normal: list[tuple[Optional[dt.time], Optional[dt.time]]] = [
    (dt.time.min, dt.time(hour=9, minute=30)),
    (dt.time(hour=16), dt.time.max)]
_time_range_when_market_close_extend: list[tuple[Optional[dt.time], Optional[dt.time]]] = [
    (dt.time.min, dt.time(hour=4)),
    (dt.time(hour=20), dt.time.max)]

bc_all = BackupCondition()
bc_weekday = BackupCondition(on_weekday=_weekday)

bc_weekday_during_market_open_normal = BackupCondition(
    on_weekday=_weekday, on_time_range=_time_range_during_market_open_normal)
bc_weekday_during_market_open_extend = BackupCondition(
    on_weekday=_weekday, on_time_range=_time_range_during_market_open_extend)

bc_weekday_when_market_close_normal = BackupCondition(
    on_weekday=_weekday, on_time_range=_time_range_when_market_close_normal)
bc_weekday_when_market_close_extend = BackupCondition(
    on_weekday=_weekday, on_time_range=_time_range_when_market_close_extend)

bc_weekday_after_market_close_normal = BackupCondition(
    on_weekday=_weekday, on_time_range=_time_range_after_market_close_normal)
bc_weekday_after_market_close_extend = BackupCondition(
    on_weekday=_weekday, on_time_range=_time_range_after_market_close_extend)


bc_friday_after_market_close_normal = BackupCondition(
    on_weekday={4}, on_time_range=_time_range_after_market_close_normal)
bc_friday_after_market_close_extend = BackupCondition(
    on_weekday={4}, on_time_range=_time_range_after_market_close_extend)

bc_weekday_before_market_open_normal = BackupCondition(
    on_weekday=_weekday, on_time_range=_time_range_before_market_open_normal)
bc_weekday_before_market_open_extend = BackupCondition(
    on_weekday=_weekday, on_time_range=_time_range_before_market_open_extend)


@dataclass(kw_only=True)
class BaseTask:

    # Task Info
    backup_freq: BackupFrequency
    backup_cond: BackupCondition
    
    download_switch: int
    name: Optional[str] = None

    # History Input
    interval: Interval = Interval.DAY
    # Below argument used to generate period / start / end automatically
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

    def get_args(self, run_datetime: dt.datetime):
        return {
            'download_switch': self.download_switch,
            'download_full_text_news': self.download_full_text_news,
            'interval': self.interval,
            **self._parse_history_range_args(run_datetime),
            'history_extra_options': self.history_extra_options
        }

    def _parse_history_range_args(self, run_datetime) -> dict:

        _period, _start, _end = None, None, None
        # Only history download need to parse the date range
        if self.download_switch & DownloadSwitch.HISTORY:

            # The priority is
            # ( Start | End > Period ) > Past Days

            # Sanity check for input
            if all(map(lambda x: x is None, [self.period, self.start, self.end])):
                if self.past_days < 0:
                    raise ValueError("Given input are invalid: period, start and end are all None and past_days is invalid")

            # start / end has higher priority than period
            if (self.start is None
                and self.end is None
                and self.past_days < 0
                and self.period is not None):  # only Period is not None
                _period = self.period
            else:
                # All others cases:
                # - start and end are both None, 
                # Guarantee the largest coverage of result
                if self.end is not None:
                    _end = self._parse_input_date(self.end, dt.time.max)
                else:
                    _end = dt.datetime.combine(
                        run_datetime.date() - dt.timedelta(days=self.end_day_offset),
                        dt.time.max)

                if self.start is not None:
                    _start = self._parse_input_date(self.start, dt.time.min)
                else:
                    _start = dt.datetime.combine(
                        _end - dt.timedelta(days=self.past_days),
                        dt.time.min)

        return {
            'period': _period,
            'start': _start,
            'end': _end
        }

    def _parse_input_date(self, date: str | dt.datetime | dt.date, time: Optional[dt.time] = None) -> dt.datetime:

        if isinstance(date, dt.datetime):
            return date
        elif isinstance(date, dt.date):
            _time = time or dt.time.min 
            return dt.datetime.combine(date, _time)

        elif isinstance(date, str):
            _date = pd.to_datetime(date).to_pydatetime()

            if _date == dt.datetime.combine(_date.date(), dt.time.min):
                _time = time or dt.time.min 
                return dt.datetime.combine(_date, _time)

        raise ValueError(f'Given input {date} is not valid')

@dataclass(kw_only=True)
class HistoryTask(BaseTask):

    download_switch: int = DownloadSwitch.HISTORY


@dataclass(kw_only=True)
class IntraDayHistoryTask(HistoryTask):
    past_days: int = 0 # Default to backup current days' data
    backup_freq: BackupFrequency = BackupFrequency.DAILY
    backup_cond: BackupCondition = bc_weekday_after_market_close_extend

    history_extra_options: HistoryExtraOptions = HistoryExtraOptions(prepost=True)


@dataclass(kw_only=True)
class IntraDayHistoryTaskCrypto(HistoryTask):
    """Since Crypto is trading 24/7, we need to define a special rule"""
    
    past_days: int = 0 # Default to backup current days' data
    end_day_offset: int = 1 # Today's data will always be incomplete
    backup_freq: BackupFrequency = BackupFrequency.DAILY
    backup_cond: BackupCondition = bc_all

    history_extra_options: HistoryExtraOptions = HistoryExtraOptions(prepost=True)
