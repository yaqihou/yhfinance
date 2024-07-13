"""Load from configuration and generate jobs
"""

import datetime as dt
import logging
import pathlib
import pandas as pd

from .task_config import TICKER_CONFIGS, HistoryTask, IntraDayHistoryTask, BaseTask, TickerConfig
from .db_messenger import DBMessenger as DB
from .defs import JobSetup, DownloadSwitch, JobStatus

logger = logging.getLogger("yfinance-backup.job_generator")

class JobGenerator:

    def __init__(self, ticker_configs=TICKER_CONFIGS):
        self._jobs = []
        self.ticker_configs: list[TickerConfig] = ticker_configs
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
                    raise ValueError("")

            # start / end has higher priority than period
            if task.start is None and task.end is None and task.past_days < 0 and task.period is not None:  # only Period is not None
                _period = task.period
            else:
                # All others cases:
                # - start and end are both None, 
                # Guarantee the largest coverage of result
                _end = dt.datetime.combine(pd.to_datetime(task.end or self.run_datetime.date()), dt.time.max)
                _start = dt.datetime.combine(pd.to_datetime(task.start or _end - dt.timedelta(days=task.past_days)), dt.time.min)

        return {
            'period': _period,
            'start': _start,
            'end': _end
        }

    def _gen_job(self,
               ticker_name,
               ticker_type,
               task: BaseTask | HistoryTask | IntraDayHistoryTask) -> JobSetup:
        """Generate the job spec for the given task"""

        with DB() as db:
            df = db.read_sql(f"""
            SELECT COUNT(1)
            FROM meta_runLog
            WHERE task_name = '{task.name}'
                AND run_status = {JobStatus.SUCCESS}
                AND run_date = '{self.run_datetime.date()}'
            """)

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

    def _need_run(self, task) -> bool:
        """Test if the given task need to be run"""

        with DB() as db:
            df = db.read_sql(f"""
            SELECT MAX(run_datetime)
            FROM meta_runLog
            WHERE task_name = '{task.name}'
                AND run_status = 1
            """)

        ret = False
        if df.empty or df.iloc[0, 0] is None:
            logger.debug("There is no successfull previous runLog for task %s", task.name)
            ret = True
        else:
            if (self.run_datetime - df.iloc[0, 0]) > task.backup_freq:
                logger.debug("Last run is outside wait window for task %s", task.name)
                ret = True

        return ret

    def create_jobs(self):

        for ticker_config in self.ticker_configs:
            logger.debug("Found %d tasks defined for Ticker %s (%s)",
                         len(ticker_config.tasks), ticker_config.ticker_name, ticker_config.ticker_type.value)

            _new_jobs = []
            for task in filter(self._need_run, ticker_config.tasks):
                _new_jobs.append(
                    self._gen_job(ticker_config.ticker_name, ticker_config.ticker_type, task)
                )

            logger.info("Generated %d new jobs for Ticker %s", len(_new_jobs), ticker_config.ticker_name)
            self._jobs += _new_jobs

        return self.jobs

    @property
    def jobs(self):
        return self._jobs
