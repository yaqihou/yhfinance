import datetime as dt

from .defs import TickerType, UserConfig
from .tasks_preset import TaskPreset



TICKER_CONFIGS = [
    UserConfig(
        ticker_name = 'TQQQ',
        ticker_type = TickerType.ETF,
        added_date = dt.date(2024, 7, 8),
        # tasks=[TASK_OPTION, TASK_NEWS, TASK_DAY] + TASKS_INTRADAY
        tasks=TaskPreset.all_tasks
    ),
]
