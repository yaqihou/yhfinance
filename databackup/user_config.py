import datetime as dt

from .defs import TickerType, UserConfig
from .tasks_factory import TaskPreset

task_preset_factory = TaskPreset()

etf_tickers = ['TQQQ', 'SQQQ', 'QQQ',
               'FAS', 'FAZ',
               'BOIL',
               'SOXX', 'USD',
               'SPY', 'IVV',
               'NVDL', 'NVDX', 'NVD',
               'JET', 'JETU', 'JETD',
               'IBIT']

mag7_stock_tickers = ['AAPL', 'TSLA', 'GOOG', 'MSFT', 'META', 'AMZN', 'NVDA']
bank_stock_tickers = ['WFC', 'JPM', 'BAC', 'MS', 'HSBC', 'GS', 'C', 'UBS', 'TD']
airlines_stock_tickers = ['LUV', 'AAL', 'DAL', 'UAL', 'ALK', 'SKYW']
idx_tickers = ['^IXIC', '^DJI', '^GSPC', '^RUT']
# TODO rates_tickers = []
crypto_tickers = ['BTC-USD', 'ETH-USD', 'USDT-USD']

TICKER_CONFIGS = [
    # *[UserConfig(
    #     ticker_name = ticker,
    #     ticker_type = TickerType.ETF,
    #     tasks=task_preset_factory.all_tasks
    # ) for ticker in etf_tickers],
    # *[UserConfig(
    #     ticker_name = ticker,
    #     ticker_type = TickerType.STOCK,
    #     tasks=task_preset_factory.all_tasks
    # ) for ticker in mag7_stock_tickers],
    # *[UserConfig(
    #     ticker_name = ticker,
    #     ticker_type = TickerType.STOCK,
    #     tasks=task_preset_factory.all_tasks
    # ) for ticker in bank_stock_tickers],
    # *[UserConfig(
    #     ticker_name = ticker,
    #     ticker_type = TickerType.STOCK,
    #     tasks=task_preset_factory.all_tasks
    # ) for ticker in airlines_stock_tickers],
    # *[UserConfig(
    #     ticker_name = ticker,
    #     ticker_type = TickerType.Index,
    #     tasks=task_preset_factory.all_tasks
    # ) for ticker in idx_tickers],
    *[UserConfig(
        ticker_name = ticker,
        ticker_type = TickerType.Crypto,
        tasks=task_preset_factory.all_tasks_crypto
    ) for ticker in crypto_tickers],
]
