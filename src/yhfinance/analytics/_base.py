
import pandas as pd

from .const import Col

# TODO - add more price cols on the fly
#        Median. The Median Price of every bar for the specified period - (High + Low) / 2.
#        Typical. The sum of High, Low, and Close prices divided by 3 for the specified period- (High + Low + Close) / 3.
#        OHLC average. The arithmetical mean of High, Low, Open, and Close prices for the specified period - (Open + High + Low + Close) / 4.

class _OHLCBase:

    def __init__(
            self, df: pd.DataFrame,
            tick_col: str = Col.Date.name):

        # TODO - add some clearn up / sanity check for OHLC data
        self._df_raw: pd.DataFrame = df.copy()
        self._df: pd.DataFrame = df.copy()

        self.tick_col = tick_col
        self._df = self._df.sort_values(by=self.tick_col)

    @property
    def _base_columns(self):
        return self._df_raw.columns
        
    @property
    def df(self):
        return self._df

    def get_result(self):
        return self.df
