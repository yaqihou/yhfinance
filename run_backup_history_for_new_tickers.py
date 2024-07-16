
from databackup.logger import MyLogger
logger_setup = MyLogger()
logger_setup.setup()
logger = logger_setup.logger

from databackup.tasks_factory import TaskForNewTicker
from databackup.defs import TickerType, UserConfig
from databackup.db_utils import DBFetcher
from databackup.data_backup import DataBackup

from watchlist import DEFAULT_WATCHLIST

# Check if the ticker is really new
# fetcher = DBFetcher()

# df = db_messenger.read_sql(f'''
# SELECT COUNT(1)
# FROM {TableName.History.PRICE_TABLE_MAPPING['1m']}
# WHERE ticker_name = '{ticker_name}'
#     AND ticker_type = '{ticker_type.value}'
# ''')

# if df.empty or df.iloc[0, 0] > 0:
#     print(f'There are existing data for {ticker_name}, do you want to continue?')

# all_ticker_list = [(config.ticker_name, config.ticker_type) for config in USER_TICKER_CONFIGS]
all_ticker_list = [('NVDQ', TickerType.ETF)]

for ticker_name, ticker_type in all_ticker_list:
    task_factory = TaskForNewTicker(ticker_name, ticker_type)
    tasks = task_factory.get_all_tasks()

    config = [UserConfig(
        ticker_name = ticker_name,
        ticker_type = ticker_type,
        tasks = tasks
    )]

    data_backuper = DataBackup(config)
    data_backuper.run()
