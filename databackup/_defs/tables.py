

class OptionTableName:

    EXPIRATIONS = 'data_options_expirations'
    CALLS = 'data_options_calls'
    PUTS = 'data_options_puts'
    UNDERLYINGS = 'data_options_underlyings'

    @classmethod
    def to_list(cls):
        return [cls.EXPIRATIONS, cls.CALLS, cls.PUTS, cls.UNDERLYINGS]


class NewsTableName:

    CONTENT = 'data_news_content'
    RELATION =  'data_news_relation'

    @classmethod
    def to_list(cls):
        return [cls.CONTENT, cls.RELATION]

class HistoryTableName:

    ARGS =  'data_history_args'
    
    PRICE_TABLE_MAPPING = {
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

    METADATA_TABLE_MAPPING = {
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

    @classmethod
    def to_list(cls):
        return (list(cls.PRICE_TABLE_MAPPING.values())
                + list(cls.METADATA_TABLE_MAPPING.values())
                + [cls.ARGS])


class HolderTableName:

    MAJOR = 'data_holder_majorHolders'
    INSTITUTIONAL = 'data_holder_institutionalHolders'
    MUTUAL_FUND = 'data_holder_mutualfundHolders'
    INSIDER_TRANSACTION = 'data_holder_insiderTransactions'
    INSIDER_PURCHASE = 'data_holder_insiderPurchases'
    INSIDER_ROSTER= 'data_holder_insiderRosterHolders'

    @classmethod
    def to_list(cls):
        return [cls.MAJOR, cls.INSTITUTIONAL, cls.MUTUAL_FUND, cls.INSIDER_TRANSACTION, cls.INSIDER_PURCHASE, cls.INSIDER_ROSTER]
        

class FinancialTableName:

    INCOME_STMP       = 'data_financial_is'
    QTR_INCOME_STMP   = 'data_financial_qtrIs'

    BALANCE_SHEET     = 'data_financial_bs'
    QTR_BALANCE_SHEET = 'data_financial_qtrBs'

    CASHFLOW          = 'data_financial_cf'
    QTR_CASHFLOW      = 'data_financial_qtrCf'

    EARNINGS_DATES    = 'data_financial_earningsDates'

    @classmethod
    def to_list(cls):
        return [cls.INCOME_STMP, cls.QTR_INCOME_STMP,
                cls.BALANCE_SHEET, cls.QTR_BALANCE_SHEET,
                cls.CASHFLOW, cls.QTR_CASHFLOW,
                cls.EARNINGS_DATES]


class RatingTableName:

    RECOMMENDATIONS = 'data_rating_recommendations'
    RECOMMENDATIONS_SUMMARY = 'data_rating_recommendationsSummary'
    UPGRADES_DOWNGRADES = 'data_rating_upgradesDowngrades'
    
    @classmethod
    def to_list(cls):
        return [cls.RECOMMENDATIONS, cls.RECOMMENDATIONS_SUMMARY, cls.UPGRADES_DOWNGRADES]


class MetaTableName:

    run_log = 'meta_runLog'
    tickers = 'meta_tickers'
    tasks = 'meta_tasks'

    @classmethod
    def to_list(cls):
        return [cls.run_log, cls.tasks, cls.tickers]


class TableName:

    Option = OptionTableName
    News = NewsTableName
    History = HistoryTableName
    Holder = HolderTableName
    Info = 'data_info'
    Financial = FinancialTableName
    Rating = RatingTableName
    Meta = MetaTableName

    @classmethod
    def to_list(cls, include_meta: bool = False):

        ret = []
        for t in [cls.Option, cls.News, cls.History, cls.Holder, cls.Financial, cls.Rating]:
            ret += t.to_list()

        ret.append(cls.Info)

        if include_meta:
            ret += cls.Meta.to_list()

        return ret


class MetaTableDefinition:

    run_log = f"""
    CREATE TABLE "{TableName.Meta.run_log}" (
        "ticker_name"	TEXT,
        "ticker_type"	TEXT,
        "run_date"	DATE,
        "run_datetime"	TIMESTAMP,
        "run_intraday_version"	INTEGER,
        "run_status"	INTEGER,
        "task_name"	TEXT
    );"""

    tickers = f"""
    CREATE TABLE "{TableName.Meta.tickers}" (
        "ticker_id"	INTEGER NOT NULL UNIQUE,
        "ticker_name"	TEXT,
        "ticker_type"	TEXT,
        PRIMARY KEY("ticker_id" AUTOINCREMENT)
    );"""

    tasks = f"""
    CREATE TABLE "{TableName.Meta.tasks}" (
        "task_id"	INTEGER NOT NULL UNIQUE,
        "task_name"	TEXT,
        "backup_freq"	TEXT,
        ""
        PRIMARY KEY("ticker_id" AUTOINCREMENT)
    );"""
