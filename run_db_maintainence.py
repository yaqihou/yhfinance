from databackup.logger import MyLoggerSetup
logger_setup = MyLoggerSetup()
logger_setup.setup()
logger = logger_setup.logger
from databackup.db_utils import DBMaintainer


db_maintainer = DBMaintainer()

db_maintainer.maintain_unique_entries(dryrun=True)
