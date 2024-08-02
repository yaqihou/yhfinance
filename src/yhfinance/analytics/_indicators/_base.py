
from numpy.typing import ArrayLike
import pandas as pd

import abc
from ..ohlc_cols import Col, ColName
from ..ohlc_data import OHLCData, OHLCDataBase

from ._indicators_mixin import *


class _BaseIndicator(_PriceColMixin, OHLCDataBase, abc.ABC):

    # TODO - need to implement for each indicator
    _category: str = "undefined"
    _abbrev: str = 'undefined'
    _fullname: str

    def __init__(self, data: OHLCData, price_col: ColName = Col.Close):
        super().__init__(data, price_col=price_col)

        self.calc()

    def calc(self):
        _df = self._calc()
        self._add_result_safely(_df)

    @abc.abstractmethod
    def _calc(self) -> pd.DataFrame:
        """Implement the calculation and return the results wanted to add to self._df"""

    @abc.abstractmethod
    def make_addplot(self, plotter_args: dict, *args, **kwargs) -> list[dict]:
        """Return a list of panel definition returned by mpf.make_addplot"""

    @property
    @abc.abstractmethod
    def values(self):
        """Return the final result(s)as np.array"""

    @property
    def need_new_panel_num(self) -> bool:
        return False

    @property
    def category(self) -> str:
        return self.category
    
    # @property
    # @abc.abstractmethod
    # def default_panel(self) -> int:
    #     """Return -1 if need a new panel, otherwise"""

    def _add_result_safely(self, df: pd.DataFrame):
        """Add the result df to self._df without causing duplicated columns.
        Need to make sure df contrains only the key column (tick_col) and desired results
        """
        drop_cols = []
        for col in df.columns:
            if col == self.tick_col:  continue
            if col in self._df.columns:
                # print(f'Warning: found existing RSI result col {col}, dropping it')
                drop_cols.append(col)
        
        if drop_cols:
            self._df.drop(columns=drop_cols, inplace=True)
        self._df = self._df.merge(df, how='left', on=self.tick_col)
