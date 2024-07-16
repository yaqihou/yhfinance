from collections import Counter
from tabulate import tabulate


from yhfinance.db_utils import DB
from yhfinance.logger import MyLogger

from yhfinance.const.databackup import JobSetup, JobStatus, UserConfig

from .job_generator import JobGenerator
from .data_puller import TickerPuller

logger = MyLogger('main')


class DataBackup:

    def __init__(self, ticker_configs: list[UserConfig]):
        self.job_generator = JobGenerator(ticker_configs=ticker_configs)
        self.db = DB()

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

        self.db.add_job_status(job, status.value)

        return status



