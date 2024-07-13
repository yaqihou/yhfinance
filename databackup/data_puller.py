
import abc
import datetime as dt
from functools import wraps
from typing import Optional

from pandas.core.arrays import period
import yfinance as yf
import pandas as pd

from requests import Session
from requests_cache import CacheMixin, SQLiteCache
from requests_ratelimiter import LimiterMixin, MemoryQueueBucket
from pyrate_limiter import Duration, RequestRate, Limiter

import requests_cache



from .defs import JobSetup, DownloadSwitch
from .data_container import *

import logging

# TODO - need better exception handler

# TODO - add log for downloaded results
logger = logging.getLogger("yfinance-backup.data-puller")


class DownloadLog:

    def __init__(self, fmt_str, *logger_args, **logger_kwargs):
        self.fmt_str = fmt_str
        self.logger_args = logger_args
        self.logger_kwargs = logger_kwargs
        
    def __enter__(self):
        logger.info('Download started: ' + self.fmt_str, *self.logger_args, **self.logger_kwargs)

    def __exit__(self, exc_type, exc_value, exc_traceback):
        logger.info('Download finished: ' + self.fmt_str, *self.logger_args, **self.logger_kwargs)


# class CachedLimiterSession(CacheMixin, LimiterMixin, Session):
#     pass

class LimiterSession(LimiterMixin, Session):
    pass

class TickerPuller(abc.ABC):

    # session = CachedLimiterSession(
    #     limiter=Limiter(RequestRate(2, Duration.SECOND*5)),  # max 2 requests per 5 seconds
    #     bucket_class=MemoryQueueBucket,
    #     backend=SQLiteCache("yfinance.cache"),
    # )
    session = LimiterSession(
        limiter=Limiter(RequestRate(5, Duration.SECOND*5)),  # max 2 requests per 5 seconds
        bucket_class=MemoryQueueBucket,
    )
    session.headers['User-agent'] = 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/126.0.0.0 Safari/537.36'

    def __init__(self, job: JobSetup):
        self.job = job

        self.ticker: yf.Ticker = yf.Ticker(self.job.ticker_name, session=self.session)
        self._df_result: pd.DataFrame

        self._history: Optional[HistoryData] = None
        self._holder: Optional[HolderData] = None
        self._info: Optional[InfoData] = None
        self._news: Optional[NewsData] = None
        self._financial: Optional[FinancialData] = None
        self._recommendation: Optional[RecommendationData] = None
        self._option: Optional[OptionData] = None

        self._data: DataContainer = DataContainer(job)
    
    def get_data(self):
        return self._data
        
    @property
    def data(self):
        return self.get_data()

    def download(self):

        if self.job.download_switch & DownloadSwitch.HISTORY:
            with DownloadLog(
                    "history data for %s [%s]", self.job.ticker_name, self.job.history_range):
                self._download_history()

        if self.job.download_switch & DownloadSwitch.FINANCIAL:
            with DownloadLog("financial data for %s", self.job.ticker_name):
                self._download_financial()

        if self.job.download_switch & DownloadSwitch.HOLDER:
            with DownloadLog("holder data for %s", self.job.ticker_name):
                self._download_holders()

        if self.job.download_switch & DownloadSwitch.RECOMMENDATION:
            with DownloadLog("recommendations data for %s", self.job.ticker_name):
                self._download_recommendations()

        if self.job.download_switch & DownloadSwitch.NEWS:
            with DownloadLog("news data for %s", self.job.ticker_name):
                self._download_news()

        if self.job.download_switch & DownloadSwitch.INFO:
            with DownloadLog("info data for %s", self.job.ticker_name):
                self._download_info()

        if self.job.download_switch & DownloadSwitch.OPTION:
            with DownloadLog("options data for %s", self.job.ticker_name):
                self._download_option()

        self._data = DataContainer(
            job=self.job,
            history=self._history,
            info=self._info,
            news=self._news,
            holder=self._holder,
            financial=self._financial,
            recommendation=self._recommendation,
            option=self._option
        )

    def _download_financial(self):

        self._financial = FinancialData(
            income_stmt=self.ticker.income_stmt,
            qtr_income_stmt=self.ticker.quarterly_income_stmt,
            #
            balance_sheet=self.ticker.balance_sheet,
            qtr_balance_sheet=self.ticker.quarterly_balance_sheet,
            #
            cashflow=self.ticker.cashflow,
            qtr_cashflow=self.ticker.quarterly_cashflow,
            #
            earnings_dates=self.ticker.earnings_dates
        )

    def _download_holders(self):

        self._holder = HolderData(
            major_holders = self.ticker.major_holders,
            institutional_holders = self.ticker.institutional_holders,
            mutualfund_holders = self.ticker.mutualfund_holders,
            insider_transactions = self.ticker.insider_transactions,
            insider_purchases = self.ticker.insider_purchases,
            insider_roster_holders = self.ticker.insider_roster_holders
        )
        
    def _download_recommendations(self):

        self._recommendation = RecommendationData(
            recommendations=self.ticker.recommendations,
            recommendations_summary=self.ticker.recommendations_summary,
            upgrades_downgrades=self.ticker.upgrades_downgrades)

    def _download_option(self):
        """Download and organize the options results
        """
        expirations = self.ticker.options

        calls = {}
        puts = {}
        underlying = {}
        for exp_date in expirations:
            logger.info('Downloading option data for %s @ %s', self.job.ticker_name, exp_date)
            options = self.ticker.option_chain(exp_date)
            calls[exp_date] = options.calls
            puts[exp_date] = options.puts
            underlying[exp_date] = options.underlying

        self._option = OptionData(
            expirations = expirations,
            calls = calls,
            puts = puts,
            underlying = underlying
        )

    # TODO - full text parsing is needed
    def _download_news(self):

            self._news = NewsData(
                news_list=[News(news_dict) for news_dict in self.ticker.news])

            if self.job.download_full_text_news:
                print("Warning: full text fetching has not been implemented")

    def _download_history(self):
        """Download data related with history prices"""

        # Need a dry run to get meta data correctly, seems to be a bug from yfinance
        _ = self.ticker.history(interval="1d", period='1d')
        _ = self.ticker.history_metadata

        # Some special treatment
        _df_history = self.ticker.history(**self.job.history_args).reset_index()
        if 'Date' in _df_history.columns:  # the index is Date for day history and Datetime for intraday
            _df_history.rename(columns={'Date': 'Datetime'}, inplace=True)
        _df_history['Date'] = _df_history['Datetime'].apply(lambda x: x.date())
        
        self._history = HistoryData(
            history=_df_history,
            # actions=self.ticker.actions,
            args=self.job.history_args,
            metadata=self.ticker.history_metadata)

    def _download_info(self):
        """Download extra info"""
        self._info = InfoData(info=self.ticker.info)
