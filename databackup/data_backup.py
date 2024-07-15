import pandas as pd

from .logger import MyLogger
from .job_generator import JobGenerator
from .data_puller import TickerPuller
from .defs import *
from .tasks_factory import TaskPreset
from .db_utils import DBMessenger as DB
from .user_config import TICKER_CONFIGS

logger = MyLogger.getLogger('backup')


class DataBackup:

    def __init__(self, ticker_configs: list[UserConfig] = TICKER_CONFIGS):
        self.job_generator = JobGenerator(ticker_configs=ticker_configs)
        self.db = DB()

    def run(self):

        self.jobs = self.job_generator.create_jobs()
        for job in self.jobs:
            self._run_pulling_job(job)


    def _run_pulling_job(self, job: JobSetup):

        puller = TickerPuller(job)

        status: int = JobStatus.FAIL
        try:
            puller.download()
        except Exception as e:
            logger.error("Encounter error in data pulling", exc_info=e)
        else:
            status = JobStatus.SUCCESS_PULL
        
        if status is JobStatus.SUCCESS_PULL:
            try:
                puller.data.dump()
            # TODO create custom exception for data dumping
            except Exception as e:
                logger.error("Encounter error in data dumping", exc_info=e)
            else:
                status = JobStatus.SUCCESS

        self._update_job_status_to_db(job, status)

    def _update_job_status_to_db(self, job: JobSetup, status: int):
        _df = pd.DataFrame.from_dict({
            'run_status': [status],
            **{k: [v] for k, v in job.metainfo.items()}
        })

        with self.db.conn as conn:
            _df.to_sql(TableName.Meta.run_log, conn, if_exists='append', index=False)


