
from copy import deepcopy
import pandas as pd

from .const import Col

# TODO - add more price cols on the fly
#        Median. The Median Price of every bar for the specified period - (High + Low) / 2.
#        Typical. The sum of High, Low, and Close prices divided by 3 for the specified period- (High + Low + Close) / 3.
#        OHLC average. The arithmetical mean of High, Low, Open, and Close prices for the specified period - (Open + High + Low + Close) / 4.
class OHLCData:

    def __init__(self, df: pd.DataFrame, tick_col: str = Col.Date.name):

        # TODO - add some clearn up / sanity check for OHLC data
        self.df_raw: pd.DataFrame = df.copy()
        self.df: pd.DataFrame = df.copy()

        self.tick_col = tick_col
        self.df.sort_values(by=self.tick_col, inplace=True)

    @property
    def _base_columns(self):
        return self.df_raw.columns
        
    def copy(self):
        return deepcopy(self)

    def __add__(self, obj):

        if isinstance(obj, pd.DataFrame):
            _obj_df = obj
        elif isinstance(obj, OHLCData):
            _obj_df = obj.df
        else:
            raise ValueError(f"Could not add a OHLCData with {obj.__class__.__name__}")

        common_cols = list(set(self.df.columns) & set(_obj_df.columns))
        _df = self.df.merge(_obj_df, on=common_cols, how='left')

        return OHLCData(df=_df, tick_col=self.tick_col)


class OHLCDataBase:

    def __init__(self, data: OHLCData):
        self._data: OHLCData = OHLCData(data.df, data.tick_col)

    @property
    def tick_col(self) -> str:
        return self._data.tick_col

    @property
    def _df(self) -> pd.DataFrame:
        return self._data.df

    @_df.setter
    def _df(self, val):
        assert isinstance(val, (pd.DataFrame))
        self._data.df = val

    @property
    def df(self) -> pd.DataFrame:
        """A read-only for user"""
        return self._df

    @property
    def _df_raw(self) -> pd.DataFrame:
        return self._data.df_raw

    def get_result(self):
        return self._df
