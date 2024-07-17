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

        logger.debug('Databackup initiated')

    def run(self):

        self.jobs: list[JobSetup] = self.job_generator.create_jobs()

        self.run_res = []
        for job in self.jobs:
            self.run_res.append(self._run_pulling_job(job))

        self._report()

    def _report(self):

        logger.info('Finished for %d jobs in total', len(self.jobs))

        counter = Counter(self.run_res)
        _summary = [[k.name, v] for k, v in counter.items()]
        logger.info("The summary is as below:\n%s", tabulate(_summary, headers=['Status', 'Count']))

        _msg = []
        for job, status in zip(self.jobs, self.run_res):
            if status is not JobStatus.SUCCESS:
                _msg.append(f'   - {job.ticker_name} ({job.ticker_type.name}): {job.task.name}')
            
        if _msg:
            logger.error('The followings failed during the run\n%s', '\n'.join(_msg))


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



