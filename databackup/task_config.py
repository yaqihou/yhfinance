from dataclasses import dataclass
import datetime as dt
from typing import Optional

from .defs import TickerType, BackupFrequency, Interval
from .defs import BaseTask, HistoryTask, IntraDayHistoryTask, TickerConfig
from .defs import DownloadSwitch as DS


TASK_OPTION = BaseTask(
    name = "intraday_options",
    backup_freq = BackupFrequency.ONE_THIRD_DAY,
    download_switch = DS.OPTION
)

TASK_NEWS = BaseTask(
    name = "intraday_news",
    backup_freq = BackupFrequency.HOUR_6,
    download_switch = DS.NEWS
)

TASK_NEWS_FULLTEXT = BaseTask(
    name = "intraday_news_fulltext",
    backup_freq = BackupFrequency.HOUR_6,
    download_switch = DS.NEWS,
    download_full_text_news = True
)

TASK_FINANCIAL = BaseTask(
    name = "monthly_financial",
    backup_freq = BackupFrequency.MONTHLY,
    download_switch = DS.FINANCIAL
)

TASK_RECOMMENDATION = BaseTask(
    name = "weekly_recommendation",
    backup_freq=BackupFrequency.WEEKLY,
    download_switch = DS.RECOMMENDATION
)

TASK_HOLDER = BaseTask(
    name = "weekly_holder",
    backup_freq=BackupFrequency.WEEKLY,
    download_switch = DS.HOLDER
)

TASK_M01  = IntraDayHistoryTask(interval=Interval.MIN_1, name="intraday_m01_history")
TASK_M02  = IntraDayHistoryTask(interval=Interval.MIN_2, name="intraday_m02_history")
TASK_M05  = IntraDayHistoryTask(interval=Interval.MIN_5, name="intraday_m05_history")
TASK_M15 = IntraDayHistoryTask(interval=Interval.MIN_15, name="intraday_m15_history")
TASK_M30 = IntraDayHistoryTask(interval=Interval.MIN_30, name="intraday_m30_history")
TASK_M60 = IntraDayHistoryTask(interval=Interval.MIN_60, name="intraday_m60_history")
TASK_M90 = IntraDayHistoryTask(interval=Interval.MIN_90, name="intraday_m90_history")

TASKS_INTRADAY = [TASK_M01, TASK_M02, TASK_M05, TASK_M15, TASK_M30, TASK_M60, TASK_M90]

TASK_DAY = HistoryTask(
    name="daily_price_weekly_backup",
    past_days=6, # Including today
    interval=Interval.DAY,
    backup_freq=BackupFrequency.WEEKLY
)

# TASK_NEWS_FULLTEXT
TASKS_ALL = [TASK_OPTION, TASK_NEWS, TASK_FINANCIAL, TASK_RECOMMENDATION, TASK_DAY] + TASKS_INTRADAY

ETF_TASKS = [TASK_OPTION, TASK_DAY] + TASKS_INTRADAY

TICKER_CONFIGS = [
    TickerConfig(
        ticker_name = 'TQQQ',
        ticker_type = TickerType.ETF,
        added_date = dt.date(2024, 7, 8),
        # tasks=[TASK_OPTION, TASK_NEWS, TASK_DAY] + TASKS_INTRADAY
        tasks=[TASK_DAY, TASK_M01]
    ),
]
