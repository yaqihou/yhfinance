"""Load from configuration and generate jobs
"""

import datetime as dt
import pandas as pd

from typing import Optional


from yhfinance.const.databackup import BackupFrequency, JobStatus, JobSetup
from yhfinance.const.databackup import HistoryTask, IntraDayHistoryTask, BaseTask, UserConfig
from yhfinance.const.db import TableName
from yhfinance.const.tickers import TickerType

from yhfinance.logger import MyLogger
from yhfinance.db_utils import DBFetcher

logger = MyLogger.getLogger("job-gen")

class JobGenerator:

    def __init__(self, ticker_configs: list[UserConfig]):
        self._jobs = []
        self.ticker_configs: list[UserConfig] = ticker_configs
        self.run_datetime = dt.datetime.today()
        self.fetcher = DBFetcher()
  
    def _gen_job(self,
               ticker_name: str,
               ticker_type: TickerType,
               task: BaseTask | HistoryTask | IntraDayHistoryTask) -> JobSetup:
        """Generate the job spec for the given task"""

        sql = f"""
        SELECT COUNT(1) AS cnt
        FROM [{TableName.Meta.run_log}]
        WHERE task_name = '{task.name}'
            AND run_status = {JobStatus.SUCCESS.value}
            AND run_date = '{self.run_datetime.date()}'
            AND ticker_name = '{ticker_name}'
            AND ticker_type = '{ticker_type.value}'
        """
        df = self.fetcher.read_sql(sql)

        if df.empty:  intraday_ver = 1
        else:
            intraday_ver = df.iloc[0, 0] + 1

        args = {
            'ticker_name'  : ticker_name,
            'ticker_type'  : ticker_type,
            'run_datetime' : self.run_datetime,
            'run_intraday_version': intraday_ver,
            'task'         : task,
            **task.get_args(self.run_datetime),
        }

        job = JobSetup(**args)
        self.fetcher.db.add_job_status(job, JobStatus.INIT.value)
        return job

    def _has_enough_gap_since_last_run(self, task,
                                       buffer_time: dt.timedelta = dt.timedelta(minutes=15)):

        df = self.fetcher.read_sql(f"""
        SELECT MAX(run_datetime)
        FROM [{TableName.Meta.run_log}]
        WHERE task_name = '{task.name}'
            AND run_status = 1
        """)

        ret = False
        if df.empty or df.iloc[0, 0] is None:
            logger.debug("There is no successful runs in the log for task %s", task.name)
            ret = True
        else:
            last_run_time = pd.to_datetime(df.iloc[0, 0])
            if ((self.run_datetime - last_run_time)
                > (task.backup_freq.value - buffer_time)):
                logger.debug("Last run is outside wait window for task %s: %s - %s > %s with buffer %s",
                             task.name, str(self.run_datetime), str(last_run_time),
                             str(task.backup_freq.value), str(buffer_time))
                ret = True
            else:
                logger.debug("Task %s is too new to initiate", task.name)

        return ret

    def _check_backup_conditions(self, task) -> list[bool]:
        """Return the list of results if each backup condition is met"""

        res = task.backup_cond.check(self.run_datetime)
        for k, v in res.items():
            if not v:
                logger.debug("Condition %s is not met as of %s", k, str(self.run_datetime))

        return list(res.values())
    
    def _is_valid_task(self, task: BaseTask) -> bool:
        """Test if the given task need to be run"""

        if task.backup_freq is BackupFrequency.AD_HOC:
            logger.info('Task %s is AD_HOC task, always added', task.name)
            return True

        satisfy_backup_freq = self._has_enough_gap_since_last_run(task)
        satisfy_conditions = self._check_backup_conditions(task)

        ret = all([satisfy_backup_freq, *satisfy_conditions])
        if not ret:
            logger.info("Task %s will NOT be added", task.name)
        else:
            logger.info("Task %s meet all backup specs and will be added", task.name)

        return ret

    def create_jobs(self):

        for ticker_config in self.ticker_configs:
            logger.debug("Found %d tasks defined for Ticker %s (%s)",
                         len(ticker_config.tasks), ticker_config.ticker_name, ticker_config.ticker_type.value)

            _new_jobs = [
                self._gen_job(ticker_config.ticker_name, ticker_config.ticker_type, task)
                for task in ticker_config.tasks
                if self._is_valid_task(task)
            ]

            logger.info("Generated %d new jobs for Ticker %s", len(_new_jobs), ticker_config.ticker_name)
            self._jobs += _new_jobs

        logger.info("Generated %d jobs in total for %d Tickers", len(self.jobs), len(self.ticker_configs))
        return self.jobs

    @property
    def jobs(self):
        return self._jobs
