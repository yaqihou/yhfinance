
import abc
import datetime as dt
from typing import Optional

from tabulate import tabulate
import pandas as pd
import json
from dataclasses import dataclass

from yhfinance.const.databackup import JobSetup
from yhfinance.const.db import TableName

from yhfinance.db_utils import DB
from yhfinance.logger import MyLogger

logger = MyLogger("datadump")

@dataclass(kw_only=True)
class BaseData(abc.ABC):

    db = DB()
    
    def dump(self, job: JobSetup):
        logger.info('Dumping %s for Ticker %s into database', self.__class__.__name__, job.ticker_name)

        self._dump_to_db(job)

    def _dump_to_db(self, job: JobSetup):
        """Save data to database"""

        self._df_for_db_list: list[tuple[pd.DataFrame, str]] = self._prepare_df_for_db(job)

        if not self._df_for_db_list:
            logger.info('Found empty %s to dump for Ticker %s', self.__class__.__name__, job.ticker_name)
        else:
            self._log_before_dump_to_db()

        for _df, _tbl_name in self._df_for_db_list:
            self.db.add_df(_df, _tbl_name, if_exists='append')
            
    @abc.abstractmethod
    def _prepare_df_for_db(self, job: JobSetup) -> list[tuple[pd.DataFrame, str]]:
        """Return a list of (DataFrame, table_name) to be dumped to DB"""

    def _log_before_dump_to_db(self):
        """Logging related information to log, assuming the """
        pass

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
                    (df_expirations, TableName.Option.EXPIRATIONS),
                    (df_calls, TableName.Option.CALLS),
                    (df_puts, TableName.Option.PUTS),
                    (df_underlyings, TableName.Option.UNDERLYINGS),
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
            (df_news, TableName.News.CONTENT),
            (df_relation, TableName.News.RELATION),
        ]
        return [(x, y) for x, y in ret if not x.empty]
            

@dataclass(kw_only=True)
class HistoryData(BaseData):
    history_raw: pd.DataFrame
    # actions -- actions are included in 
    # actions: pd.DataFrame  
    #
    args: dict
    metadata: dict

    @property
    def empty(self):
        return self.history_raw is None or self.history_raw.empty

    def __post_init__(self):

        if self.empty:
            logger.debug("Found empty history DataFrame, skip post-processing")
        else:
            # Add special treatment to the raw data
            _df_history = self.history_raw.reset_index().copy()
            if 'Date' in _df_history.columns:  # the index is Date for day history and Datetime for intraday
                _df_history.rename(columns={'Date': 'Datetime'}, inplace=True)

            # NOTE - or use .normalize() but may be better to just keep the date() 
            _df_history['Date'] = _df_history['Datetime'].apply(lambda x: x.date())

            _interval = self.args['interval']
            _is_intraday = _interval[-1] in {'m', 'h'}

            # add the indicator for period type
            # NOTE - the trading periods 
            _df_history = self._apply_trading_period_type(_df_history, _is_intraday, use_metadata=True)

            # don't save action for intraday history
            if _is_intraday:
                _cols = [x for x in _df_history.columns
                        if x.upper() not in {"DIVIDENDS", "STOCK SPLITS", "CAPITAL GAINS"}]
            else:
                _cols = _df_history.columns

            self.history = _df_history[_cols].copy()

    def _apply_trading_period_type(self, df, is_intraday: bool,
                                   use_metadata: bool = True
                                   ) -> pd.DataFrame:

        df['period_type'] = 'regular'
        
        if not (is_intraday and self.metadata.get('hasPrePostMarketData', False)):
            logger.info('The history do not have pre/post market data')
        else:
            if (not use_metadata
                or (self.metadata.get('tradingPeriods') is None
                    and self.metadata.get('currentTradingPeriod') is None)):

                if not use_metadata:
                    logger.debug('use_metadata is overriden to False')
                else:
                    logger.debug('meta data do not have tradingPeriods or currentTradingPeriod field')
                    logger.debug('History metadata %s', str(self.metadata))
                df = self._apply_trading_period_type_without_metadata(df)
            else:
                if self.metadata.get('tradingPeriods') is not None:
                    logger.info('Parsing period_type using metadata in tradingPeriods')
                    df = self._apply_trading_period_type_with_trading_periods(df)

                elif self.metadata.get('currentTradingPeriod') is not None:
                    logger.debug('History metadata %s', str(self.metadata))
                    logger.info('Parsing period_type using metadata in currentTradingPeriod')
                    df = self._apply_trading_period_type_with_current_trading_period(df)

                else:
                    # DEBUG
                    raise ValueError('Something must be wrong')

        return df

    def _apply_trading_period_type_with_trading_periods(self, df) -> pd.DataFrame:

        df_meta = self.metadata['tradingPeriods'].reset_index(names=['tmp_key'])
        logger.debug('Metadata tradingPeriods %s', str(df_meta.to_csv()))

        df['tmp_key'] = df['Datetime'].dt.normalize()
        if len(df_meta.columns) > 3:
            logger.debug('Found more than 3 columns in tradingPeriod metadata: %s',
                         ', '.join(df_meta.columns))

        df = df.merge(df_meta[['tmp_key', 'start', 'end']], on='tmp_key')
        df.loc[df['Datetime'] < df['start'] , 'period_type'] = 'pre'
        df.loc[df['Datetime'] >= df['end'] , 'period_type'] = 'post'

        df = df.drop(columns=['tmp_key', 'start', 'end'])

        return df

    def _apply_trading_period_type_with_current_trading_period(self, df) -> pd.DataFrame:

        _unique_dates = df['Date'].unique()
        if len(_unique_dates) > 1:
            # This is possible - for example, if we pull with period = 'max', then there will be
            # no tradingPeriods, while currentTradingPeriod is always a dict for current day
            logger.warn('There are more than 1 unique dates in dataframe: %s without tradingPeriods, skip parsing',
                        ', '.join(map(str, _unique_dates)))
            return df

        # Below is from yfinance.util
        tz = self.metadata["exchangeTimezoneName"]
        _parsed_md = {}
        for m in ["regular", "pre", "post"]:
            if (m in self.metadata["currentTradingPeriod"]
                and isinstance(self.metadata["currentTradingPeriod"][m]["start"], int)
                ):
                _parsed_md[m] = {}
                for t in ["start", "end"]:
                    _parsed_md[m][t] = pd.to_datetime(
                        self.metadata["currentTradingPeriod"][m][t],
                        unit='s',
                        utc=True).tz_convert(tz)

        if 'regular' in _parsed_md:
            _start = _parsed_md['regular']['start']
            _end = _parsed_md['regular']['end']

            logger.debug('Found trading period as [%s, %s)', _start, _end)

            df.loc[df['Datetime'] < _start, 'period_type'] = 'pre'
            df.loc[df['Datetime'] >= _end, 'period_type'] = 'post'

        else:
            logger.warn('No regular defined in currentTradingPeriod, skip parsing')
                

        return df
        

    def _apply_trading_period_type_without_metadata(self, df) -> pd.DataFrame:
        logger.warning('Parsing period_type without metadata has not been implemented')
        return df

    def _prepare_df_for_db(self, job: JobSetup) -> list[tuple[pd.DataFrame, str]]:

        if self.empty:  return []

        # Only select OLHC data for intraday history
        _interval = self.args['interval']
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
            (self.history, TableName.History.PRICE_TABLE_MAPPING[_interval]),
            (_df_args, TableName.History.ARGS),
            (_df_meta, TableName.History.METADATA_TABLE_MAPPING[_interval]),
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
        ret = [
            (self.major_holders          , TableName.Holder.MAJOR),
            (self.institutional_holders  , TableName.Holder.INSTITUTIONAL),
            (self.mutualfund_holders     , TableName.Holder.MUTUAL_FUND),
            (self.insider_transactions   , TableName.Holder.INSIDER_TRANSACTION),
            (self.insider_purchases      , TableName.Holder.INSIDER_PURCHASE),
            (self.insider_roster_holders , TableName.Holder.INSIDER_ROSTER)
        ]
        return [(self._add_job_metainfo_cols(x, job), y) for x, y in ret if x is not None and not x.empty]


@dataclass
class InfoData(BaseData):

    info: dict

    def _prepare_df_for_db(self, job: JobSetup):

        _df = pd.DataFrame.from_dict({
            'info_json': [json.dumps(self.info)]
        })
        return [(self._add_job_metainfo_cols(_df, job), TableName.INFO)]


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

        tmp_lst = [
                (self.income_stmt       , TableName.Financial.INCOME_STMP),
                (self.qtr_income_stmt   , TableName.Financial.QTR_INCOME_STMP),
                (self.balance_sheet     , TableName.Financial.BALANCE_SHEET),
                (self.qtr_balance_sheet , TableName.Financial.QTR_BALANCE_SHEET),
                (self.cashflow          , TableName.Financial.CASHFLOW),
                (self.qtr_cashflow      , TableName.Financial.QTR_CASHFLOW)
        ]

        ret = []
        for df, tbl_name in tmp_lst:
            if df is not None and not df.empty:
                _df = df.transpose().reset_index(names='report_date')
                ret.append(
                    (self._add_job_metainfo_cols(_df, job), tbl_name)
                )

        if self.earnings_dates is not None and not self.earnings_dates.empty:
            df = self.earnings_dates.reset_index()
            ret.append(
                (self._add_job_metainfo_cols(df, job), TableName.Financial.EARNINGS_DATES)
            )

        return ret

        
@dataclass
class RatingData(BaseData):

    recommendations: pd.DataFrame 
    recommendations_summary: pd.DataFrame 
    upgrades_downgrades: pd.DataFrame 

    def _prepare_df_for_db(self, job: JobSetup) -> list[tuple[pd.DataFrame, str]]:

        tmp = [
            (self.recommendations, TableName.Rating.RECOMMENDATIONS),
            (self.recommendations_summary, TableName.Rating.RECOMMENDATIONS_SUMMARY)
        ]

        ret = []
        for df, tbl_name in tmp:

            if not df.empty:
                ret.append(
                    (self._add_job_metainfo_cols(df, job), tbl_name)
                )
                
        # Need reset_index
        if not self.upgrades_downgrades.empty:
            df = self.upgrades_downgrades.reset_index()
            ret.append(
                (self._add_job_metainfo_cols(df, job), TableName.Rating.UPGRADES_DOWNGRADES)
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
    rating: Optional[RatingData] = None
    option: Optional[OptionData] = None

    def dump(self):

        for data in [self.history,
                      self.info,
                      self.news,
                      self.holder,
                      self.financial,
                      self.rating,
                      self.option]:
            if data is not None:
                data.dump(self.job)
