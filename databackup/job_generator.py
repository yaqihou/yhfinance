"""Load from configuration and generate jobs
"""

import datetime as dt
import logging
import pathlib
import pandas as pd


from .db_messenger import DBMessenger as DB
from .defs import JobSetup, DownloadSwitch, JobStatus, TableName, TickerType
from .defs import HistoryTask, IntraDayHistoryTask, BaseTask, UserConfig
from .user_config import TICKER_CONFIGS

logger = logging.getLogger("yfinance-backup.job_generator")

class JobGenerator:

    def __init__(self, ticker_configs=TICKER_CONFIGS):
        self._jobs = []
        self.ticker_configs: list[UserConfig] = ticker_configs
        self.run_datetime = dt.datetime.today()

    def _parse_history_range_args(self, task) -> dict:

        _period, _start, _end = None, None, None
        # Only history download need to parse the date range
        if task.download_switch & DownloadSwitch.HISTORY:

            # The priority is
            # ( Start | End > Period ) > Past Days

            # Sanity check for input
            if all(map(lambda x: x is None, [task.period, task.start, task.end])):
                if task.past_days < 0:
                    logger.error("Past days is not set")
                    raise ValueError("Given input are invalid: period, start and end are all None and past_days is invalid")

            # start / end has higher priority than period
            if (task.start is None
                and task.end is None
                and task.past_days < 0
                and task.period is not None):  # only Period is not None
                _period = task.period
            else:
                # All others cases:
                # - start and end are both None, 
                # Guarantee the largest coverage of result
                if task.end is not None:
                    _end = pd.to_datetime(task.end)
                else:
                    _end = dt.datetime.combine(
                        self.run_datetime.date() - dt.timedelta(days=task.end_day_offset),
                        dt.time.max)

                if task.start is not None:
                    _start = pd.to_datetime(task.start)
                else:
                    _start = dt.datetime.combine(
                        _end - dt.timedelta(days=task.past_days),
                        dt.time.min)

        return {
            'period': _period,
            'start': _start,
            'end': _end
        }

    def _gen_job(self,
               ticker_name: str,
               ticker_type: TickerType,
               task: BaseTask | HistoryTask | IntraDayHistoryTask) -> JobSetup:
        """Generate the job spec for the given task"""

        with DB() as db:
            sql = f"""
            SELECT COUNT(1) AS cnt
            FROM [{TableName.Meta.run_log}]
            WHERE task_name = '{task.name}'
                AND run_status = {JobStatus.SUCCESS}
                AND run_date = '{self.run_datetime.date()}'
                AND ticker_name = '{ticker_name}'
                AND ticker_type = '{ticker_type.value}'
            """
            df = db.read_sql(sql)

        if df.empty:  intraday_ver = 1
        else:
            intraday_ver = df.iloc[0, 0] + 1

        args = {
            'ticker_name'  : ticker_name,
            'ticker_type'  : ticker_type,
            'run_datetime' : self.run_datetime,
            'run_intraday_version': intraday_ver,
            'task'         : task,
            **task.get_args(),
            **self._parse_history_range_args(task)
        }

        return JobSetup(**args)

    def _has_enough_gap_since_last_run(self, task):

        with DB() as db:
            df = db.read_sql(f"""
            SELECT MAX(run_datetime)
            FROM [{TableName.Meta.run_log}]
            WHERE task_name = '{task.name}'
                AND run_status = 1
            """)

        ret = False
        if df.empty or df.iloc[0, 0] is None:
            logger.debug("There is no successfull previous runLog for task %s", task.name)
            ret = True
        else:
            if (self.run_datetime - pd.to_datetime(df.iloc[0, 0])) > task.backup_freq.value:
                logger.debug("Last run is outside wait window for task %s", task.name)
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
    
    def _is_valid_task(self, task) -> bool:
        """Test if the given task need to be run"""

        satisfy_backup_freq = self._has_enough_gap_since_last_run(task)
        satisfy_conditions = self._check_backup_conditions(task)

        ret = all([satisfy_backup_freq, *satisfy_conditions])
        if not ret:
            logger.info("Task %s is not needed to be run at this moment", task.name)
        else:
            logger.info("Task %s meet all backup specs and is added", task.name)

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

        return self.jobs

    @property
    def jobs(self):
        return self._jobs
