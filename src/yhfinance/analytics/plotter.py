
from typing import Optional, Literal

import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.axes
import matplotlib.figure
import mplfinance as mpf

import pandas as pd

from .const import Col, ColName


_T_TYPE_LITERAL = Literal['candle', 'renko', 'pnf', 'ohlc', 'line', 'hnf']
class OHLCMpfPlotter:
    _T_TYPE_LITERAL = _T_TYPE_LITERAL

    def __init__(
            self,
            tick_col: str,  # the columnt used as index
            tight_layout: bool = True,
            noshow: bool = False):

        self.tick_col = tick_col
        self.tight_layout = tight_layout
        self.noshow = noshow

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        if self.tight_layout:
            plt.tight_layout()

        if not self.noshow:
            plt.show()

    @property
    def mpf_plot_base_args(self):
        return {'returnfig': True}

    def _plot(self, df: pd.DataFrame, **kwargs):
        return mpf.plot(df, **kwargs, **self.mpf_plot_base_args) 

    def plot_basic(
            self,
            df: pd.DataFrame,
            type: _T_TYPE_LITERAL = 'candle',
            volume: bool = True,
            mav=(3,),
            **kwargs
    ):

        return self._plot(df, volume=volume, type=type, mav=mav, **kwargs)


    # TODO - add historical recession overlay
    def plot_macd(self):
        return self._plot(df, **kwargs, **self.mpf_plot_base_args) 


class OHLCMultiFigurePlotter(OHLCMpfPlotter):
    """Collection of different plot types for OHLC-type data
    """

    def __init__(
            self,
            tick_col: str,
            tight_layout: bool = True,
            noshow: bool = False,
            #
            ncols: int = 1,
            nrows: int = 1,
            figscale: float = 1,
    ):
        super().__init__(tick_col, tight_layout, noshow)

        self.ncols = ncols
        self.nrows = nrows
        # Useful for subfigures
        self.figscale = figscale
        self.figsize: tuple[float, float] = tuple(map(
            lambda x: x * self.figscale,
            (4 * ncols + 0.4 * (ncols - 1), 3 * nrows + 0.4 * (nrows - 1))
        ))

        self.fig = plt.figure(figsize=self.figsize)
        self.subfigs = self.fig.subfigures(nrows, ncols)
        self.current_subifg = self.subfigs[0][0]
        
    def select_subfig(self, subfig_idx: int):

        _idx_col = subfig_idx % self.ncols
        _idx_row = subfig_idx // self.ncols
        self.current_subifg = self.subfigs[_idx_row][_idx_col]

    def set_subfig_suptitle(self, title):
        self.current_subifg.suptitle(title)

    @property
    def mpf_plot_base_args(self):
        return {**super().mpf_plot_base_args, 'subfig': self.current_subifg}
