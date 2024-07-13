
from databackup.logger import MyLoggerSetup
logger_setup = MyLoggerSetup()
logger_setup.setup()
logger = logger_setup.logger

import datetime as dt
import pandas as pd

from databackup.job_generator import JobGenerator
from databackup.data_puller import TickerPuller
from databackup.defs import *
from databackup.task_config import *
from databackup.db_messenger import DBMessenger as DB

df = pd.DataFrame.from_dict({
    'ticker_name': ['TEST'],
    'ticker_type': ['ETF'],
    'run_date': [dt.date.today()],
    'run_datetime': [dt.datetime.today()],
    'run_intraday_version': [0],
    'run_status': [0],
    'task_name': 'test_task'
})
with DB().conn as conn:
    df.to_sql('meta_runLog', conn, if_exists='replace', index=False)

ad_hoc_task = IntraDayHistoryTask(
    interval=Interval.MINUTE,
    backup_freq=BackupFrequency.AD_HOC,
    download_switch=DownloadSwitch.RECOMMENDATION,
    start='2024-07-12',
    end='2024-07-12'
)

task = TASK_OPTION
# task.start = '2024-07-12'

job_generator = JobGenerator(
[
    TickerConfig(
        ticker_name = 'MSFT',
        ticker_type = TickerType.STOCK,
        added_date = dt.date(2024, 7, 8),
        tasks=[task]
    ),
    TickerConfig(
        ticker_name = 'TQQQ',
        ticker_type = TickerType.ETF,
        added_date = dt.date(2024, 7, 8),
        tasks=[task]
    ),
    TickerConfig(
        ticker_name = 'BTC-USD',
        ticker_type = TickerType.Crypto,
        added_date = dt.date(2024, 7, 8),
        tasks=[task]
    ),
    TickerConfig(
        ticker_name = '^IXIC',
        ticker_type = TickerType.Index,
        added_date = dt.date(2024, 7, 8),
        tasks=[task]
    ),
]
)
jobs = job_generator.create_jobs()


for job in jobs:

    puller = TickerPuller(job)

    status = False
    try:
        puller.download()
    except Exception as e:
        logger.error("Encounter error in data pulling", exc_info=e)
    else:
        status = True
        puller.data.dump()
    
