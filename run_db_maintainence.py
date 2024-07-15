from databackup.logger import MyLogger
MyLogger.setup()
logger = MyLogger.logger
from databackup.db_utils import DBMaintainer


db_maintainer = DBMaintainer()

# db_maintainer.maintain_unique_entries(dryrun=True)
# db_maintainer._drop_all_tables(include_meta=True)
