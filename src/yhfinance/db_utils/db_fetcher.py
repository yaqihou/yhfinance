import pandas as pd

from yhfinance.const.db import TableName
from yhfinance.logger import MyLogger

from . import DB, DBConfig

# TODO - move to a centralized place
logger = MyLogger(DBConfig.LOGGER_NAME)

class DBFetcher:

    def __init__(self, db_name: str = DBConfig.DB_NAME):
        self._db_name = db_name
        self.db = DB(db_name)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        return

    @property
    def conn(self):
        return self.db.conn
    
    def read_sql(self, sql: str) -> pd.DataFrame:
        logger.debug("Fecthing df using the following query:\n%s", sql)
        # TODO - add exception treatment
        # TODO - need to check if the columns are consistent
        return pd.read_sql(sql, self.conn)

    # --------------------------------
    # Create a bunch of shortcuts to load data
    # TODO - considering if we need to factor them into a separate class
