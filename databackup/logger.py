import sys
import logging


class MyLoggerSetup:

    default_formatter = logging.Formatter(
        fmt='%(asctime)s | %(levelname)s | %(message)s',
        datefmt='%m/%d/%Y %I:%M:%S %p')

    def __init__(self):

        self.logger = logging.getLogger("yfinance-backup")
        _ = [self.logger.removeHandler(hdlr) for hdlr in self.logger.handlers]

    def setup(self):
        self.logger.addHandler(self.hdlr_stdout)
        self.logger.addHandler(self.hdlr_file)
        self.logger.setLevel(logging.DEBUG)

    @property
    def hdlr_stdout(self):
        hdlr = logging.StreamHandler(sys.stdout)
        hdlr.setFormatter(self.default_formatter)
        hdlr.setLevel(logging.DEBUG)

        return hdlr
        
    @property
    def hdlr_file(self):
        hdlr = logging.FileHandler('finance-data-backup.log')
        hdlr.setFormatter(self.default_formatter)
        hdlr.setLevel(logging.INFO)
        return hdlr

