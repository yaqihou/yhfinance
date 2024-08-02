
import pandas as pd

import mplfinance as mpf

from ..ohlc_cols import Col, ColName
from ..ohlc_data import OHLCData
from ..ohlc_processor import OHLCInterProcessor

from ._indicators_mixin import *
from ._base import _BaseIndicator


class IndTrueRange(_BaseIndicator):

    def __init__(self,
                 data: OHLCData,
                 price_col: ColName = Col.Close):
        super().__init__(data, price_col=price_col)

    def _calc(self) -> pd.DataFrame:
        _df = OHLCInterProcessor(self._data, tick_offset=-1)._df_offset

        # TR = max[(H-L), abs(H-Cp), abs(L-Cp)]
        _df[Col.Ind.TrueRange] = pd.concat([
            _df[Col.High.cur] - _df[Col.Low.cur],
            (_df[Col.High.cur] - _df[Col.Close.sft]).abs(),
            (_df[Col.Low.cur] - _df[Col.Close.sft]).abs()
            ], axis=1).max(axis=1)

        return _df

    def make_addplot(self, plotter_args: dict, *args, **kwargs) -> list[dict]:
        raise NotImplementedError()
    
    @property
    def values(self):
        return self.df[Col.Ind.TrueRange].values
