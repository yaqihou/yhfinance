
import datetime as dt

from databackup._defs.tickers import HistoryExtraOptions

from .logger import MyLogger

logger = MyLogger.getLogger('task-factory')

from .defs import BackupFrequency
from .defs import Interval, TickerType, Period
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
    def intraday_crypto_hist_tasks(self) -> list[IntraDayHistoryTask]:
        return [
            self.INTRADAY_CRYPTO_HIST_M01,
            self.INTRADAY_CRYPTO_HIST_M02,
            self.INTRADAY_CRYPTO_HIST_M05,
            self.INTRADAY_CRYPTO_HIST_M15,
            self.INTRADAY_CRYPTO_HIST_M30,
            self.INTRADAY_CRYPTO_HIST_M60,
            self.INTRADAY_CRYPTO_HIST_M90
        ]

    @property
    def intraday_tasks(self) -> list[BaseTask]:
        return [self.OPTION_3X_PER_DAY, self.NEWS_4X_PER_DAY] + self.intraday_hist_tasks

    @property
    def intraday_crypto_tasks(self) -> list[BaseTask]:
        return [
            # Crypto Currency do not need option
            # self.OPTION_3X_PER_DAY,
            self.NEWS_4X_PER_DAY] + self.intraday_crypto_hist_tasks

    @property
    def weekly_tasks(self) -> list[BaseTask]:
        return [self.RATING_WEEKLY, self.HOLDER_WEEKLY, self.INFO_WEEKLY]

    @property
    def all_tasks(self) -> list[BaseTask]:
        return self.intraday_tasks + self.weekly_tasks + [self.DAY_HIST_DAILY, self.FINANCIAL_MONTHLY]

    @property
    def all_tasks_crypto(self) -> list[BaseTask]:
        return self.intraday_crypto_tasks + [self.DAY_HIST_DAILY]


class TaskForNewTicker:
    """Use to generate AdHoc """

    def __init__(self, ticker_name: str, ticker_type: TickerType):
        self.ticker_name: str = ticker_name
        self.ticker_type: TickerType = ticker_type

    def _gen_intraday_tasks(self, interval: Interval) -> list[BaseTask]:
        """Return ghe limitation for 
        """

        # Limits on intraday data:
        #     • 1m = max 7 days within last 30 days
        #     • 60m, 1h = max last 730 days
        #     • else up to 90m = max 60 days

        today = dt.datetime.today()
        _market_open = dt.datetime.combine(today.date(), dt.time(hour=4))
        _market_close = dt.datetime.combine(today.date(), dt.time(hour=20))


        end_date = today.date()
        if self.ticker_type is TickerType.Crypto:
            end_date -= dt.timedelta(days=1)
            logger.warning(
                "Crypto market is 24/7, shift end_date to previous date %s", str(end_date))

        else:
            if today.weekday() in {0, 1, 2, 3, 4}:
                if today >= _market_open and today < _market_close:
                    end_date -= dt.timedelta(days=1)
                    logger.warning(
                        "The extended market is still open now, intraday history may be incomplete, using previous date %s", str(end_date))

        if interval.value == '1m':
            max_time_window = 29
        elif interval.value == '60m' or interval.value == '1h':
            max_time_window = 729
        else:
            max_time_window = 59

        start_date = today.date() - dt.timedelta(days=max_time_window)

        tasks: list[BaseTask] = []
        if interval.value == '1m':
            # For minute-level history, we can only fetch 7 days each time
            _start_date = start_date
            part_num = 0

            # Each window (step) would be 8 days, but this is fine as we have weekends and holidays
            # The trading days cannot exceed 7 days in this case
            step = dt.timedelta(days=6)
            while _start_date <= end_date:
                _end_date = _start_date + step  # no need to cap here
                tasks.append(
                    BaseTask(
                        name=f"intraday_{interval.value}_history_adhoc_pt{part_num}_{today}",
                        #
                        backup_freq=BackupFrequency.AD_HOC,
                        backup_cond=bc_all,
                        #
                        download_switch=DS.HISTORY,
                        #
                        interval=interval,
                        start=dt.datetime.combine(_start_date, dt.time.min),
                        end=dt.datetime.combine(_end_date, dt.time.max),
                        history_extra_options=HistoryExtraOptions(prepost=True)
                    )
                )
                _start_date = _end_date + dt.timedelta(days=1)
                part_num += 1

        else:
            tasks.append(
                BaseTask(
                    name=f"intraday_{interval.value}_history_adhoc_{today}",
                    #
                    backup_freq=BackupFrequency.AD_HOC,
                    backup_cond=bc_all,
                    #
                    download_switch=DS.HISTORY,
                    #
                    interval=interval,
                    start=dt.datetime.combine(start_date, dt.time.min),
                    end=dt.datetime.combine(end_date, dt.time.max),
                    history_extra_options=HistoryExtraOptions(prepost=True)
                )
            )
                
        return tasks

    def get_intraday_tasks(self) -> list[BaseTask]:

        intraday_tasks: list[BaseTask] = []

        for interval in Interval.get_all_intraday_intervals():
            intraday_tasks += self._gen_intraday_tasks(interval)

        return intraday_tasks
        
    def _gen_day_tasks(self, interval: Interval) -> list[BaseTask]:
        """Return ghe limitation for 
        """

        assert not interval.is_intraday, 'Given interval is intrday'

        tasks: list[BaseTask] = [
            BaseTask(
                name=f"day_{interval.value}_history_adhoc_{dt.date.today()}",
                #
                backup_freq=BackupFrequency.AD_HOC,
                backup_cond=bc_all,
                #
                download_switch=DS.HISTORY,
                #
                interval=interval,
                period=Period.MAX
            )
        ]
                
        return tasks

    def get_day_tasks(self) -> list[BaseTask]:

        return self._gen_day_tasks(Interval.DAY)

    @property
    def all_tasks(self) -> list[BaseTask]:
        return self.get_intraday_tasks() + self.get_day_tasks()
