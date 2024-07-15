
from databackup.logger import MyLogger
logger_setup = MyLogger()
logger_setup.setup()
logger = logger_setup.logger

import datetime as dt
import pandas as pd

from databackup.job_generator import JobGenerator
from databackup.data_puller import TickerPuller
from databackup.defs import *
from databackup.tasks_factory import TaskPreset
from databackup.db_utils import DBMessenger as DB


task = TaskPreset.INTRADAY_CRYPTO_HIST_M01
task.backup_freq = BackupFrequency.AD_HOC

job_generator = JobGenerator()
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
    
        _df = pd.DataFrame.from_dict(
            {
                'run_status': [JobStatus.SUCCESS],
                **{k: [v] for k, v in job.metainfo.items()}
            }
        )

        with DB().conn as conn:
            _df.to_sql(TableName.Meta.run_log, conn, if_exists='append', index=False)

    
