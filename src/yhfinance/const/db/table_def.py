
from .table_name import TableName

class MetaTableDefinition:

    run_log = f"""
    CREATE TABLE "{TableName.Meta.run_log}" (
        "ticker_name"	TEXT,
        "ticker_type"	TEXT,
        "run_date"	DATE,
        "run_datetime"	TIMESTAMP,
        "run_intraday_version"	INTEGER,
        "run_status"	INTEGER,
        "task_name"	TEXT
    );"""

    tickers = f"""
    CREATE TABLE "{TableName.Meta.tickers}" (
        "ticker_id"	INTEGER NOT NULL UNIQUE,
        "ticker_name"	TEXT,
        "ticker_type"	TEXT,
        PRIMARY KEY("ticker_id" AUTOINCREMENT)
    );"""

    tasks = f"""
    CREATE TABLE "{TableName.Meta.tasks}" (
        "task_id"	INTEGER NOT NULL UNIQUE,
        "task_name"	TEXT,
        "backup_freq"	TEXT,
        ""
        PRIMARY KEY("ticker_id" AUTOINCREMENT)
    );"""
