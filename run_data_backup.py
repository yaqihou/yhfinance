from argparse import ArgumentParser

from databackup.logger import MyLogger
from databackup.data_backup import DataBackup
from user_config import USER_TICKER_CONFIGS

parser = ArgumentParser()
parser.add_argument('-l', '--log', type=str, default=None, nargs='?')

args = parser.parse_args()

MyLogger.setup(log_filename=args.log)

data_backuper = DataBackup(USER_TICKER_CONFIGS)
data_backuper.run()
