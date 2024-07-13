
import abc
import datetime as dt
import logging
from typing import Optional

import pandas as pd
import json
from dataclasses import dataclass

from .defs import JobSetup
from .db_messenger import DBMessenger as DB

logger = logging.getLogger("yfinance-backup.datadump")

@dataclass(kw_only=True)
class BaseData(abc.ABC):

    def dump(self, job: JobSetup):
        logger.info('Dumping %s for Ticker %s into database', self.__class__.__name__, job.ticker_name)

        self._dump_to_db(job)

    def _dump_to_db(self, job: JobSetup):
        """Save data to database"""

        self._df_for_db_list: list[tuple[pd.DataFrame, str]] = self._prepare_df_for_db(job)

        if not self._df_for_db_list:
            logger.info('Found empty %s to dump for Ticker %s', self.__class__.__name__, job.ticker_name)

        for _df, _tbl_name in self._df_for_db_list:
            self._dump_df_to_db(_df, _tbl_name, if_exists='append')
            
    @abc.abstractmethod
    def _prepare_df_for_db(self, job: JobSetup) -> list[tuple[pd.DataFrame, str]]:
        """Return a list of (DataFrame, table_name) to be dumped to DB"""

    @staticmethod
    def _dump_df_to_db(df, table_name, if_exists='fail'):

        logger.debug('Dumping DataFrame to %s', table_name)

        try:
            with DB().conn as conn:
                df.to_sql(table_name, conn, if_exists=if_exists, index=False)
        except Exception as e:
            logger.error('Encountered error when dumping DataFrame to %s', table_name, exc_info=e)
        else:
            logger.info('Successfully dump DataFrame into %s', table_name)

    @staticmethod
    def _flatten_df_dict(df_dict: dict[str, pd.DataFrame]) -> pd.DataFrame:
        df_lst = []
        for exp_date, _df in df_dict.items():
            _df = _df.copy()
            _df['expire'] = pd.to_datetime(exp_date)
            df_lst.append(_df)
        return pd.concat(df_lst, ignore_index=True)

    @staticmethod
    def _add_job_metainfo_cols(df, job, metainfo_cols=None):

        cols = metainfo_cols or job.metainfo.keys()

        for col in cols:
            if col not in df.columns:
                df[col] = job.metainfo[col]

        return df


@dataclass(kw_only=True)
class OptionData(BaseData):

    expirations: tuple[str, ...]
    calls: dict[str, pd.DataFrame]
    puts: dict[str, pd.DataFrame]
    underlying: dict[str, dict]

    def _prepare_df_for_db(self, job: JobSetup) -> list[tuple[pd.DataFrame, str]]:

        if self.expirations:

            df_expirations = pd.DataFrame.from_dict({
                'expire': list(map(pd.to_datetime, self.expirations))
            })
            df_calls = self._flatten_df_dict(self.calls)
            df_puts = self._flatten_df_dict(self.puts)

            _data_dict = {'expire': [], 'underlying_json': []}
            for exp, _dict in self.underlying.items():
                _data_dict['expire'].append(pd.to_datetime(exp))
                _data_dict['underlying_json'].append(json.dumps(_dict))
            df_underlyings = pd.DataFrame.from_dict(_data_dict)

            ret = [
                    (df_expirations, 'data_options_expirations'),
                    (df_calls, 'data_options_calls'),
                    (df_puts, 'data_options_puts'),
                    (df_underlyings, 'data_options_underlyings'),
            ]

            return [(self._add_job_metainfo_cols(x, job), y) for x, y in ret]
        else:
            return []
    
   
class News:

    def __init__(self, news_dict):
        self.news_dict = news_dict
        self.half_text = ""
        self.full_text = ""

    @property
    def title(self) -> str:
        return self.news_dict.get('title', "")
        
    @property
    def type(self) -> str:
        return self.news_dict.get('type', "").upper()

    @property
    def link(self) -> str:
        return self.news_dict.get('link', "")
    
    @property
    def publisher(self) -> str:
        return self.news_dict.get('publisher', "")

    @property
    def uuid(self) -> str:
        return self.news_dict.get('uuid', "")

    @property
    def provider_publish_time(self) -> dt.datetime:
        return dt.datetime.fromtimestamp(
            int(self.news_dict.get('providerPublishTime', '0')))

    @property
    def related_tickers(self) -> list[str]:
        return self.news_dict.get('relatedTickers', [])


@dataclass
class NewsData(BaseData):

    news_list: list[News]

    def _prepare_df_for_db(self, job: JobSetup) -> list[tuple[pd.DataFrame, str]]:

        # TODO - some optimizations are possible here
        data_dict = {
            'uuid': [],
            'link': [],
            'type': [],
            'title': [],
            'publisher': [],
            'publish_time': [],
        }

        relation_dict = {
            'uuid': [],
            'ticker': []
        }

        for news in self.news_list:
            data_dict['uuid'].append(news.uuid)
            data_dict['link'].append(news.link)
            data_dict['type'].append(news.type)
            data_dict['title'].append(news.title)
            data_dict['publisher'].append(news.publisher)
            data_dict['publish_time'].append(news.provider_publish_time)

            for ticker in news.related_tickers:
                relation_dict['uuid'].append(news.uuid)
                relation_dict['ticker'].append(ticker)

        df_news = pd.DataFrame.from_dict(data_dict)
        df_relation = pd.DataFrame.from_dict(relation_dict)

        ret = [
            (df_news, 'data_news_content'),
            (df_relation, 'data_news_relation'),
        ]
        return [(x, y) for x, y in ret if not x.empty]
            

@dataclass(kw_only=True)
class HistoryData(BaseData):
    history: pd.DataFrame
    # actions -- actions are included in 
    # actions: pd.DataFrame  
    #
    args: dict
    metadata: dict

    HISTORY_TABLE_MAPPING = {
        '1m': 'data_history_intra_min01',
        '2m': 'data_history_intra_min02',
        '5m': 'data_history_intra_min05',
        '15m': 'data_history_intra_min15',
        '30m': 'data_history_intra_min30',
        '60m': 'data_history_intra_min60',
        '90m': 'data_history_intra_min90',
        '1d': 'data_history_day_day1',
        '5d': 'data_history_day_day5',
        '1wk': 'data_history_day_day7',
        '1mo': 'data_history_day_mon1',
        '3mo': 'data_history_day_mon3',
    }

    HISTORY_META_TABLE_MAPPING = {
        '1m': 'data_history_meta_intra_min01',
        '2m': 'data_history_meta_intra_min02',
        '5m': 'data_history_meta_intra_min05',
        '15m': 'data_history_meta_intra_min15',
        '30m': 'data_history_meta_intra_min30',
        '60m': 'data_history_meta_intra_min60',
        '90m': 'data_history_meta_intra_min90',
        '1d': 'data_history_meta_day_day1',
        '5d': 'data_history_meta_day_day5',
        '1wk': 'data_history_meta_day_day7',
        '1mo': 'data_history_meta_day_mon1',
        '3mo': 'data_history_meta_day_mon3',
    }

    def _prepare_df_for_db(self, job: JobSetup):

        # Only select OLHC data for intraday history
        _interval = self.args['interval']
        _is_intraday = not (_interval in {'1d', '5d', '1wk', '1mo', '3mo'})
        # don't save action for intraday history
        if _is_intraday:
            _cols = [x for x in self.history.columns
                     if x.upper() not in {"DIVIDENDS", "STOCK SPLITS", "CAPITAL GAINS"}]
        else:
            _cols = self.history.columns
        _df_history = self.history[_cols].copy()

        _df_args = pd.DataFrame.from_dict({k: [v] for k,v in self.args.items()})

        # Special Treatment for history_metadata
        # Some fields does not fit the table:
        #    'tradingPeriods' key corresponds to another pd.DataFrame, ignore it
        #    'validRanges' don't need it
        #    'currentTradingPeriod': need to expand it
        _meta_dict = {}
        for k, v in self.metadata.items():

            if k == 'tradingPeriods':
                logger.debug('Ignore field tradingPeriods in history metadata')
            elif k == 'validRanges':
                logger.debug('Ignore field validRanges in history metadata')
            elif k == 'currentTradingPeriod':
                logger.debug('Expanding field currentTradingPeriod in history metadata')
                for _k2, _dct in v.items():
                    for _k3, _v3 in _dct.items():
                        _meta_dict['-'.join(['ctp', _k2, _k3])] = _v3
            else:
                _meta_dict[k] = [v]
        _df_meta = pd.DataFrame.from_dict(_meta_dict)

        ret = [
            (_df_history, self.HISTORY_TABLE_MAPPING[_interval]),
            (_df_args, 'data_history_args'),
            (_df_meta, self.HISTORY_META_TABLE_MAPPING[_interval]),
        ]

        return [(self._add_job_metainfo_cols(x, job), y) for x, y in ret]


@dataclass
class HolderData(BaseData):

    major_holders          : pd.DataFrame
    institutional_holders  : pd.DataFrame
    mutualfund_holders     : pd.DataFrame
    insider_transactions   : pd.DataFrame
    insider_purchases      : pd.DataFrame
    insider_roster_holders : pd.DataFrame
    
    def _prepare_df_for_db(self, job: JobSetup) -> list[tuple[pd.DataFrame, str]]:
        ret = []
        for holder_type, df in [
                ('majorHolders'          , self.major_holders),
                ('institutionalHolders'  , self.institutional_holders),
                ('mutualfundHolders'     , self.mutualfund_holders),
                ('insiderTransactions'   , self.insider_transactions),
                ('insiderPurchases'      , self.insider_purchases),
                ('insiderRosterHolders' , self.insider_roster_holders)]:
            if not df.empty:
                ret.append((self._add_job_metainfo_cols(df, job), f'data_holder_{holder_type}'))

        return ret


@dataclass
class InfoData(BaseData):

    info: dict

    def _prepare_df_for_db(self, job: JobSetup):

        _df = pd.DataFrame.from_dict({
            'info_json': [json.dumps(self.info)]
        })
        return [(self._add_job_metainfo_cols(_df, job), 'data_info')]


@dataclass
class FinancialData(BaseData):

    income_stmt       : pd.DataFrame
    qtr_income_stmt   : pd.DataFrame

    balance_sheet     : pd.DataFrame
    qtr_balance_sheet : pd.DataFrame

    cashflow          : pd.DataFrame
    qtr_cashflow      : pd.DataFrame

    earnings_dates    : pd.DataFrame

    def _prepare_df_for_db(self, job: JobSetup) -> list[tuple[pd.DataFrame, str]]:

        ret = []

        for report_type, df in [
                ('is', self.income_stmt),
                ('qtrIs', self.qtr_income_stmt),
                ('bs', self.balance_sheet),
                ('qtrBs', self.qtr_balance_sheet),
                ('cf', self.cashflow),
                ('qtrCf', self.qtr_cashflow)
        ]:
            if df is not None and not df.empty:
                _df = df.transpose().reset_index(names='report_date')
                ret.append(
                    (self._add_job_metainfo_cols(_df, job), f'data_financial_{report_type}')
                )

        if self.earnings_dates is not None and not self.earnings_dates.empty:
            df = self.earnings_dates.reset_index()
            ret.append(
                (self._add_job_metainfo_cols(df, job), 'data_financial_earningsDates')
            )

        return ret

        
@dataclass
class RecommendationData(BaseData):

    recommendations: pd.DataFrame 
    recommendations_summary: pd.DataFrame 
    upgrades_downgrades: pd.DataFrame 

    def _prepare_df_for_db(self, job: JobSetup) -> list[tuple[pd.DataFrame, str]]:

        ret = []
        for report_type, df in [
                ('recommendations', self.recommendations),
                ('recommendationsSummary', self.recommendations_summary),
        ]:

            if not df.empty:
                ret.append(
                    (self._add_job_metainfo_cols(df, job), f'data_rating_{report_type}')
                )
                
        # Need reset_index
        if not self.upgrades_downgrades.empty:
            df = self.upgrades_downgrades.reset_index()
            ret.append(
                (self._add_job_metainfo_cols(df, job), f'data_rating_upgradesDowngrades')
            )
                
        return ret


@dataclass
class DataContainer:

    job: JobSetup
    history: Optional[HistoryData] = None
    info: Optional[InfoData] = None
    news: Optional[NewsData] = None
    holder: Optional[HolderData] = None
    financial: Optional[FinancialData] = None
    recommendation: Optional[RecommendationData] = None
    option: Optional[OptionData] = None

    def dump(self):

        for data in [self.history,
                      self.info,
                      self.news,
                      self.holder,
                      self.financial,
                      self.recommendation,
                      self.option]:
            if data is not None:
                data.dump(self.job)
