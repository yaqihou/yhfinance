import sys
import logging

from typing import Optional


class MyLogger:

    default_formatter = logging.Formatter(
        fmt='%(asctime)s |  %(name)27s :: %(levelname)6s | %(message)s',
        datefmt='%m/%d/%Y %I:%M:%S %p')

    __PROJECT_NAME__: str | None = 'backup'
    __LOG_FILE_NAME__: str = 'finance-data-backup.log'
    # root logger 
    logger = logging.getLogger(__PROJECT_NAME__)

    @classmethod
    def getLogger(cls, name: str | None = None):
        if name is None or cls.__PROJECT_NAME__ is None:
            return logging.getLogger(name)
        else:
            return logging.getLogger('.'.join([cls.__PROJECT_NAME__, name]))
    
    @classmethod
    def setup(cls, log_filename: Optional[str] = None):
        _ = [cls.logger.removeHandler(hdlr) for hdlr in cls.logger.handlers]
        cls.logger.addHandler(cls.get_default_stdout_hdlr())
        cls.logger.addHandler(cls.get_default_file_hdlr(log_filename))
        cls.logger.setLevel(logging.DEBUG)

    @classmethod
    def get_default_stdout_hdlr(cls, level=logging.DEBUG):
        hdlr = logging.StreamHandler(sys.stdout)
        hdlr.setFormatter(cls.default_formatter)
        hdlr.setLevel(level)

        return hdlr
        
    @classmethod
    def get_default_file_hdlr(cls, log_filename: Optional[str], level=logging.INFO):
        _log_file = log_filename or cls.__LOG_FILE_NAME__
        
        hdlr = logging.FileHandler(_log_file)
        hdlr.setFormatter(cls.default_formatter)
        hdlr.setLevel(level)
        return hdlr

