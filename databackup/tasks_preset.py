
from .defs import BackupCondition, BackupFrequency
from .defs import Interval
from .defs import BaseTask, HistoryTask, IntraDayHistoryTask, IntraDayHistoryTaskCrypto
from .defs import DownloadSwitch as DS

from ._defs.tasks import bc_all,\
    bc_weekday,\
    bc_friday_after_market_close_normal,\
    bc_friday_after_market_close_extended,\
    bc_weekday_after_market_close_extended

class TaskPreset:
    
    OPTION_3X_PER_DAY = BaseTask(
        name = "options_intradayX3",
        backup_freq = BackupFrequency.ONE_THIRD_DAY,
        backup_cond = bc_weekday,
        download_switch = DS.OPTION
    )

    NEWS_4X_PER_DAY = BaseTask(
        name = "news_intradayX4",
        backup_freq = BackupFrequency.HOUR_6,
        backup_cond = bc_all,
        download_switch = DS.NEWS,
        # TODO - need implementation
        # download_full_text_news = True
    )

    FINANCIAL_MONTHLY = BaseTask(
        name = "financial_monthly",
        backup_freq = BackupFrequency.MONTHLY,
        backup_cond = bc_friday_after_market_close_extended,
        download_switch = DS.FINANCIAL
    )

    RATING_WEEKLY = BaseTask(
        name = "rating_weekly",
        backup_freq=BackupFrequency.WEEKLY,
        backup_cond = bc_friday_after_market_close_normal,
        download_switch = DS.RATING
    )

    HOLDER_WEEKLY = BaseTask(
        name = "holder_weekly",
        backup_freq=BackupFrequency.WEEKLY,
        backup_cond = bc_friday_after_market_close_normal,
        download_switch = DS.HOLDER
    )

    INTRADAY_HIST_M01  = IntraDayHistoryTask(interval=Interval.MIN_1, name="intraday_history_m01")
    INTRADAY_HIST_M02  = IntraDayHistoryTask(interval=Interval.MIN_2, name="intraday_history_m02")
    INTRADAY_HIST_M05  = IntraDayHistoryTask(interval=Interval.MIN_5, name="intraday_history_m05")
    INTRADAY_HIST_M15 = IntraDayHistoryTask(interval=Interval.MIN_15, name="intraday_history_m15")
    INTRADAY_HIST_M30 = IntraDayHistoryTask(interval=Interval.MIN_30, name="intraday_history_m30")
    INTRADAY_HIST_M60 = IntraDayHistoryTask(interval=Interval.MIN_60, name="intraday_history_m60")
    INTRADAY_HIST_M90 = IntraDayHistoryTask(interval=Interval.MIN_90, name="intraday_history_m90")

    INTRADAY_CRYPTO_HIST_M01  = IntraDayHistoryTaskCrypto(interval=Interval.MIN_1, name="intraday_crypto_history_m01")
    INTRADAY_CRYPTO_HIST_M02  = IntraDayHistoryTaskCrypto(interval=Interval.MIN_2, name="intraday_crypto_history_m02")
    INTRADAY_CRYPTO_HIST_M05  = IntraDayHistoryTaskCrypto(interval=Interval.MIN_5, name="intraday_crypto_history_m05")
    INTRADAY_CRYPTO_HIST_M15 = IntraDayHistoryTaskCrypto(interval=Interval.MIN_15, name="intraday_crypto_history_m15")
    INTRADAY_CRYPTO_HIST_M30 = IntraDayHistoryTaskCrypto(interval=Interval.MIN_30, name="intraday_crypto_history_m30")
    INTRADAY_CRYPTO_HIST_M60 = IntraDayHistoryTaskCrypto(interval=Interval.MIN_60, name="intraday_crypto_history_m60")
    INTRADAY_CRYPTO_HIST_M90 = IntraDayHistoryTaskCrypto(interval=Interval.MIN_90, name="intraday_crypto_history_m90")

    DAY_HIST_DAILY = HistoryTask(
        name="day_price_daily",
        past_days=0, # Including today
        interval=Interval.DAY,
        backup_freq=BackupFrequency.DAILY,
        backup_cond=bc_weekday_after_market_close_extended
    )

    INFO_WEEKLY = BaseTask(
        name='info_weekly',
        backup_freq=BackupFrequency.WEEKLY,
        backup_cond=bc_friday_after_market_close_normal,
        download_switch=DS.INFO
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
    def intraday_tasks(self) -> list[BaseTask]:
        return [self.OPTION_3X_PER_DAY, self.NEWS_4X_PER_DAY] + self.intraday_hist_tasks

    @property
    def weekly_tasks(self) -> list[BaseTask]:
        return [self.RATING_WEEKLY, self.HOLDER_WEEKLY, self.INFO_WEEKLY]

    @property
    def all_tasks(self) -> list[BaseTask]:
        return self.intraday_tasks + self.weekly_tasks + [self.DAY_HIST_DAILY, self.FINANCIAL_MONTHLY]
