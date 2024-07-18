

class MetaColName:

    TICKER_NAME  : str = 'ticker_name'
    TICKER_TYPE  : str = 'ticker_type'
    RUN_DATE     : str = 'run_date'
    RUN_DATETIME : str = 'run_datetime'
    INTRADAY_VER : str = 'run_intraday_version'
    TASK_NAME    : str = 'task_name'

    @classmethod
    def to_list(cls) -> list[str]:
        return [cls.TICKER_NAME, cls.TICKER_TYPE,
                cls.RUN_DATE, cls.RUN_DATETIME,
                cls.INTRADAY_VER, cls.TASK_NAME]
    
