
import pandas as pd

from .const import Col

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
