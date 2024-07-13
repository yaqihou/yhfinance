import sys
import logging
logger = logging.getLogger("yfinance-backup")
hdlr_stdout = logging.StreamHandler(sys.stdout)
hdlr_stdout.setLevel(logging.DEBUG)
logger.addHandler(hdlr_stdout)
logger.setLevel(logging.DEBUG)

import datetime as dt
import yfinance


from databackup.job_generator import JobGenerator
from databackup.data_puller import 
from databackup.task_config import *
from databackup.db_messenger import DBMessenger as DB

# db = DBMessenger()


# with db.conn as conn:
#     res = conn.execute('SELECT name FROM sqlite_master')
#     print(res.fetchone())

job_generator = JobGenerator()
ticker = TICKER_CONFIGS[0]
job_generator.create_jobs()

