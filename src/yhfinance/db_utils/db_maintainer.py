from typing import Callable, Optional

from yhfinance.logger import MyLogger
from yhfinance.const.db import TableName
from yhfinance.const.databackup import JobSetup

from . import DB, DBFetcher, DBConfig

# TODO - add this switch at module level and create testing script

logger = MyLogger(DBConfig.LOGGER_NAME)

class DBMaintainer:

    def __init__(self, db_name: str = DBConfig.DB_NAME):
        self._db_name = db_name
        self.db = DB(db_name)
        self.fetcher = DBFetcher(db_name)

    def maintain_tickers_meta(self):
        
        # Step 1 - make sure all tickers in DB are covered in meta_tickers
        # The only source of new unknown tickers are in 
        
        # Step 2 - fetch all information about each ticker

        # Step 3 - 

        return

    # TODO - def maintain_tasks_meta

    def _for_each_table(self, func: Callable, include_meta: bool = False):

        assert callable(func)

        ret = {}
        for tbl_name in TableName.to_list(include_meta=include_meta):
            if self.db._exist_table(tbl_name):
                ret[tbl_name] = func(tbl_name)
        return ret

    def _report_status(self, tbl_name):
        df = self.fetcher.read_sql(f"SELECT * FROM {tbl_name}")
        
        # Add other information we may be interested
        return df.shape, df.columns 

    def report_status(self):

        tbls_status = self._for_each_table(self._report_status)
        # TODO add a summary printout
        return
        
    def _maintain_unique_entries(
            self, tbl_name: str, dryrun: bool = False,
            ignore_meta_columns: bool = False, ignored_columns: Optional[list[str]] = None):
        
        logger.info('Cleaning up duplicate rows in table %s', tbl_name)

        df = self.fetcher.read_sql(f"SELECT * FROM {tbl_name}")

        subset = None
        if ignore_meta_columns:
            _ignored_cols = set(ignored_columns or JobSetup.get_metainfo_cols())
            subset = [x for x in df.columns if x not in _ignored_cols]
            logger.info("Will not use following columns %s", ', '.join(_ignored_cols))

        _old_shape = df.shape
        df = df.drop_duplicates(ignore_index=True, subset=subset)
        _new_shape = df.shape

        if _old_shape[0] == _new_shape[0]:
            logger.info('No cleanup neede - all entries are unique for table %s',
                        _old_shape[0], _new_shape[0], tbl_name)
        else:
            logger.info('Cleaned up duplicated rows %d -> %d in table %s',
                        _old_shape[0], _new_shape[0], tbl_name)

        if not dryrun:
            df.to_sql(tbl_name, self.db.conn, if_exists='replace', index=False)

        return

    def maintain_unique_entries(self, dryrun=False):
        self._for_each_table(
            lambda tbl_name: self._maintain_unique_entries(tbl_name, dryrun=dryrun)
        )

    def _drop_all_tables(self, include_meta: bool = False):

        with self.db.conn as conn:
            for tbl_name in TableName.to_list(include_meta=include_meta):
                logger.debug('Try to drop table %s', tbl_name)
                conn.execute(f'DROP TABLE IF EXISTS {tbl_name}')
