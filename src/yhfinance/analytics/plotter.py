
from typing import Optional, Literal

import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.axes
import matplotlib.figure
import mplfinance as mpf

import pandas as pd

from .const import Col, ColName


class OHLCPlotter:
    """Collection of different plot types for OHLC-type data
    """

    def __init__(
            self,
            tick_col: str,  # the columnt used as index
            ncols: int = 1,
            nrows: int = 1,
            figsize_scale: float = 1,
            tight_layout: bool = True,
            noshow: bool = False
    ):

        self.tick_col = tick_col

        self.ncols = ncols
        self.nrows = nrows

        self.tight_layout = tight_layout
        self.noshow = noshow
        
        self.figsize_scale = figsize_scale
        self.figsize: tuple[float, float] = tuple(map(
            lambda x: x * self.figsize_scale,
            (4 * ncols + 0.4 * (ncols - 1), 3 * nrows + 0.4 * (nrows - 1))
        ))

        # TODO - add a fallback curr_idx in case no subfig_idx given
        
        # self.fig, self.axes = plt.subplots(nrows, ncols, figsize=self.figsize)
        self.fig = plt.figure(figsize=self.figsize)
        self.subfigs = self.fig.subfigures(nrows, ncols)

    @property
    def is_single_plot(self):
        return self.ncols == 1 and self.nrows == 1

    @property
    def ax(self):
        if not self.is_single_plot:
            raise ValueError('Plotter containing multiple subfigures does not support auto ax ')
        return self.fig.gca()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        if self.tight_layout:
            plt.tight_layout()

        if not self.noshow:
            plt.show()

    def _get_subfig(self, subfig_idx: Optional[int] = None) -> matplotlib.figure.SubFigure:

        if not (self.ncols == 1 and self.nrows == 1):
            if subfig_idx is not None:
                _idx_col = subfig_idx % self.ncols
                _idx_row = subfig_idx // self.ncols
                _fig = self.subfigs[_idx_row][_idx_col]
            else:
                raise ValueError('Something wrong with ax parsing')
        else:
            _fig = self.subfigs[0][0]

        return _fig

    def plot_subfig(self,
                    subfig_idx: int,
                    df: pd.DataFrame,
                    x_col: Optional[ColName] = None,
                    y_cols: list[ColName] = [Col.Close],
                    type: Literal['simple', 'candle', 'renko', 'pnf', 'ohlc', 'line', 'hnf'] = 'simple',
                    # For other types where engine is mpf
                    mav: Optional[int | list[int] | tuple[int, ...]] = None,
                    volume: bool = False,
                    # General Arguments
                    hide_xaxis: bool = True,
                    hide_yaxis: bool = False,
                    subfig_title: str = "",
             ):

        _subfig = self._get_subfig(subfig_idx)

        if type == 'simple':
            _ax = _subfig.subplots(1, 1)
            _x = self.tick_col if x_col is None else x_col.name
            for _y_col in y_cols:
                sns.lineplot(df, x=_x, y=_y_col.name, ax=_ax)
        else:
            _df = df.set_index(self.tick_col)
            _kwargs = {'type': type}
            if volume:
                _ax, _ax_vol = _subfig.subplots(
                    2, 1, sharex=True,
                    gridspec_kw={'height_ratios': [3, 1]}
                )
                _subfig.subplots_adjust(wspace=0, hspace=0)
                _kwargs['volume'] = _ax_vol
            else:
                _ax = _subfig.subplots(1, 1)

            _kwargs['ax'] = _ax
            if mav is not None:  _kwargs['mav'] = mav

            mpf.plot(_df, **_kwargs) 

        _subfig.suptitle(subfig_title)

    # TODO - add historical recession overlay
