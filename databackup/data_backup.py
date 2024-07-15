from collections import Counter
import pandas as pd
from tabulate import tabulate

from .logger import MyLogger
from .job_generator import JobGenerator
from .data_puller import TickerPuller
from .defs import *
from .tasks_factory import TaskPreset
from .db_utils import DBFetcher

logger = MyLogger.getLogger('backup')


class DataBackup:

    def __init__(self, ticker_configs: list[UserConfig]):
        self.job_generator = JobGenerator(ticker_configs=ticker_configs)
        self.db_fetcher = DBFetcher()

    def run(self):

        self.jobs = self.job_generator.create_jobs()

        status_lst = []
        for job in self.jobs:
            status_lst.append(self._run_pulling_job(job))

        logger.info('Databack finished for %d jobs in total', len(self.jobs))
        counter = Counter(status_lst)
        _summary = [[k.name, v] for k, v in counter.items()]
        logger.info("The summary is as below:\n%s", tabulate(_summary, headers=['Status', 'Count']))

    def _run_pulling_job(self, job: JobSetup) -> JobStatus:

        puller = TickerPuller(job)

        status: JobStatus = JobStatus.FAIL
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

        self._update_job_status_to_db(job, status.value)

        return status

    def _update_job_status_to_db(self, job: JobSetup, status: int):
        _df = pd.DataFrame.from_dict({
            'run_status': [status],
            **{k: [v] for k, v in job.metainfo.items()}
        })

        with self.db_fetcher.conn as conn:
            _df.to_sql(TableName.Meta.run_log, conn, if_exists='append', index=False)


