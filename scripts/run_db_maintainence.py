from yhfinance.logger import MyLogger
from yhfinance.db_utils import DBMaintainer

MyLogger.setup()
MyLogger.setProject('db-maintain')
logger = MyLogger()

db_maintainer = DBMaintainer()

# db_maintainer.maintain_unique_entries(dryrun=True)
# db_maintainer._drop_all_tables(include_meta=True)
