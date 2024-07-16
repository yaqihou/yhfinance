import sys
import logging

from typing import Optional


class MyLogger:

    default_formatter = logging.Formatter(
        fmt='%(asctime)s |  %(name)27s :: %(levelname)6s | %(message)s',
        datefmt='%m/%d/%Y %I:%M:%S %p')

    __ROOT_NAME__: str = 'yhf'
    __CURR_PROJECT_NAME__: str  = __ROOT_NAME__
    __LOG_FILE_NAME__: str = 'yhfinance.log'

    # root logger 
    root_logger = logging.getLogger(__ROOT_NAME__)
    _reg = {}

    def __init__(self, logger_name: Optional[str] = None):
        self.logger_name: Optional[str] = logger_name

    @property
    def _logger(self):
        logger_name = self._getLoggerName(self.logger_name)
        if logger_name not in self._reg:
            self._reg[logger_name] = logging.getLogger(logger_name)

        return self._reg[logger_name]

    def info(self, msg, *args, **kwargs):
        return self._logger.info(msg, *args, **kwargs)

    def warning(self, msg, *args, **kwargs):
        return self._logger.warning(msg, *args, **kwargs)

    warn = warning

    def debug(self, msg, *args, **kwargs):
        return self._logger.debug(msg, *args, **kwargs)

    def critical(self, msg, *args, **kwargs):
        return self._logger.critical(msg, *args, **kwargs)

    def error(self, msg, *args, **kwargs):
        return self._logger.error(msg, *args, **kwargs)

    def log(self, level, msg, *args, **kwargs):
        return self._logger.log(level, msg, *args, **kwargs)

    def addHandler(self, hdlr):
        self._logger.addHandler(hdlr)

    def removeHandler(self, hdlr):
        self._logger.removeHandler(hdlr)

    def setLevel(self, level):
        return self._logger.setLevel(level)

    @property
    def handlers(self):
        return self._logger.handlers

    @property
    def name(self):
        return self._logger.name

    @property
    def level(self):
        return self._logger.level

    @property
    def parent(self):
        return self._logger.parent

    # -------------------------------------------------
    @classmethod
    def setProject(cls, proj_name: str):
        cls.__CURR_PROJECT_NAME__ = '.'.join([cls.__ROOT_NAME__, proj_name])

    @classmethod
    def _getLoggerName(cls, name: str | None = None):

        if name is None:
            return cls.__CURR_PROJECT_NAME__
        else:
            return '.'.join([cls.__CURR_PROJECT_NAME__, name])

    # TODO - clean up this part
    @classmethod
    def setup(cls, log_filename: Optional[str] = None):
        _ = [cls.root_logger.removeHandler(hdlr) for hdlr in cls.root_logger.handlers]
        cls.root_logger.addHandler(cls.get_default_stdout_hdlr())
        cls.root_logger.addHandler(cls.get_default_file_hdlr(log_filename))
        cls.root_logger.setLevel(logging.DEBUG)

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

