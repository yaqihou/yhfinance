
from databackup.logger import MyLogger
MyLogger.setup()
logger = MyLogger.logger


from databackup.data_backup import DataBackup
from user_config import USER_TICKER_CONFIGS

data_backuper = DataBackup(USER_TICKER_CONFIGS)
data_backuper.run()
