
from .defs import BackupFrequency, Interval
from .defs import BaseTask, HistoryTask, DayHistoryTask, IntraDayHistoryTask
from .defs import DownloadSwitch as DS

class TaskPreset:
    
    OPTION_3X_PER_DAY = BaseTask(
        name = "options_intradayX3",
        backup_freq = BackupFrequency.ONE_THIRD_DAY,
        download_switch = DS.OPTION
    )

    NEWS_4X_PER_DAY = BaseTask(
        name = "news_intradayX4",
        backup_freq = BackupFrequency.HOUR_6,
        download_switch = DS.NEWS,
        # TODO - need implementation
        # download_full_text_news = True
    )

    FINANCIAL_MONTHLY = BaseTask(
        name = "financial_monthly",
        backup_freq = BackupFrequency.MONTHLY,
        download_switch = DS.FINANCIAL
    )

    RATING_WEEKLY = BaseTask(
        name = "rating_weekly",
        backup_freq=BackupFrequency.WEEKLY,
        download_switch = DS.RATING
    )

    HOLDER_WEEKLY = BaseTask(
        name = "holder_weekly",
        backup_freq=BackupFrequency.WEEKLY,
        download_switch = DS.HOLDER
    )

    INTRADAY_HIST_M01  = IntraDayHistoryTask(interval=Interval.MIN_1, name="intraday_m01_history")
    INTRADAY_HIST_M02  = IntraDayHistoryTask(interval=Interval.MIN_2, name="intraday_m02_history")
    INTRADAY_HIST_M05  = IntraDayHistoryTask(interval=Interval.MIN_5, name="intraday_m05_history")
    INTRADAY_HIST_M15 = IntraDayHistoryTask(interval=Interval.MIN_15, name="intraday_m15_history")
    INTRADAY_HIST_M30 = IntraDayHistoryTask(interval=Interval.MIN_30, name="intraday_m30_history")
    INTRADAY_HIST_M60 = IntraDayHistoryTask(interval=Interval.MIN_60, name="intraday_m60_history")
    INTRADAY_HIST_M90 = IntraDayHistoryTask(interval=Interval.MIN_90, name="intraday_m90_history")

    DAY_HIST_DAILY = DayHistoryTask(
        name="day_price_daily",
        past_days=6, # Including today
        interval=Interval.DAY,
        backup_freq=BackupFrequency.WEEKLY
    )

    @property
    def intraday_hist_tasks(self) -> list[IntraDayHistoryTask]:
        return [
            self.INTRADAY_HIST_M01,
            self.INTRADAY_HIST_M02,
            self.INTRADAY_HIST_M05,
            self.INTRADAY_HIST_M15,
            self.INTRADAY_HIST_M30,
            self.INTRADAY_HIST_M60,
            self.INTRADAY_HIST_M90
        ]

    @property
    def intraday_tasks(self) -> list[BaseTask | HistoryTask | IntraDayHistoryTask]:
        return [self.OPTION_3X_PER_DAY, self.NEWS_4X_PER_DAY] + self.intraday_hist_tasks

    @property
    def weekly_tasks(self) -> list[BaseTask | HistoryTask]:
        return [self.RATING_WEEKLY, self.HOLDER_WEEKLY]

    @property
    def all_tasks(self) -> list[BaseTask | HistoryTask]:
        return self.intraday_tasks + self.weekly_tasks + [self.DAY_HIST_DAILY, self.FINANCIAL_MONTHLY]
