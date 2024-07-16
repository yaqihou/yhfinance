from argparse import ArgumentParser

from databackup.logger import MyLogger
from databackup.data_backup import DataBackup
from watchlist import DEFAULT_WATCHLIST

parser = ArgumentParser()
parser.add_argument('-l', '--log', type=str, default=None, nargs='?')

args = parser.parse_args()

MyLogger.setup(log_filename=args.log)

data_backuper = DataBackup(DEFAULT_WATCHLIST)
data_backuper.run()
