from argparse import ArgumentParser

from yhfinance.databackup.logger import MyLogger
from yhfinance.databackup.data_backup import DataBackup
from yhfinance.watchlist import DEFAULT_WATCHLIST

parser = ArgumentParser()
parser.add_argument('-l', '--log', type=str, default=None, nargs='?')

args = parser.parse_args()

MyLogger.setup(log_filename=args.log)

data_backuper = DataBackup(DEFAULT_WATCHLIST)
data_backuper.run()
