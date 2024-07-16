
import yfinance as yf

import datetime as dt

import pandas as pd

class TickerData:

    def __init__(self, ticker, session=None, proxy=None):
        self.ticker = yf.Ticker(ticker, session, proxy)

        self.df_ohlc: pd.DataFrame
        self.df_financial: pd.DataFrame
        self.run_date = dt.date.today()

    def load_ohlc(self, *args, **kwargs):
        """Load commom (level0) information
        """

    def load_financials():
        """Load commom information
        """
        
