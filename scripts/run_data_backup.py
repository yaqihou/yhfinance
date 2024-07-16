from argparse import ArgumentParser

from yhfinance.logger import MyLogger
from yhfinance.databackup.data_backup import DataBackup
from yhfinance.config.watchlist import DEFAULT_WATCHLIST

from yhfinance.db_utils import DBConfig

parser = ArgumentParser()
parser.add_argument('-l', '--log', type=str, default=None, nargs='?')
parser.add_argument('-D', '--database', type=str, nargs='?', default='')
parser.add_argument('-M', '--memory-db', action='store_true')

args = parser.parse_args()

MyLogger.setup(log_filename=args.log)
logger = MyLogger.logger

if args.memory_db:
    logger.info('Using user in-memory database')
    DBConfig.DB_NAME = ':memory:'
elif args.db:
    logger.info('Using user given database %s', args.database)
    DBConfig.DB_NAME = args.database

data_backuper = DataBackup(DEFAULT_WATCHLIST[:1])
data_backuper.run()
