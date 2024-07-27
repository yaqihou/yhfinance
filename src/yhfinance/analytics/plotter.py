
from typing import Optional, Literal

import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.axes
import matplotlib.figure
import mplfinance as mpf

import pandas as pd

from .const import Col, ColName
from .indicators import _BaseIndicator


_T_TYPE_LITERAL = Literal['candle', 'renko', 'pnf', 'ohlc', 'line', 'hnf']
class OHLCMpfPlotter:
    _T_TYPE_LITERAL = _T_TYPE_LITERAL

    def __init__(
            self,
            tick_col: str,  # the columnt used as index
            tight_layout: bool = False,
            noshow: bool = False,
            main_panel: int = 0,
            volume_panel: int = 1,
            volume: bool = False,
            figscale: Optional[float] = None,
            figratio: Optional[tuple[float, float]] = None,
            figsize: Optional[tuple[float, float]] = None,
            style: Literal['binance', 'binancedark', 'blueskies', 'brasil',
                           'charles', 'checkers', 'classic', 'default', 'ibd',
                           'kenan', 'mike', 'nightclouds', 'sas',
                           'starsandstripes', 'tradingview', 'yahoo'] = "default",
            panel_ratios: Optional[tuple[float, float] | tuple[float, ...] | list[float]] = None,
            title: Optional[str] = None,
            # Finer control in postprocess
            move_legend_outside: bool = False,
            # When there are two axis, merge it into the same legend
            merge_legend_for_each_panel: bool = True
    ):

        self.tick_col = tick_col
        self.tight_layout = tight_layout
        self.noshow = noshow

        self.main_panel: int = main_panel
        self.volume: bool = volume
        self.volume_panel: int = volume_panel

        self.figscale = figscale
        self.figratio = figratio
        self.figsize = figsize
        self.style = style
        self._panel_ratios = panel_ratios
        self.figure_title = title

        self.move_legend_outside = move_legend_outside
        self.merge_legend_for_each_panel = merge_legend_for_each_panel
        
        
        # At most 32 panels are supported
        self.reset()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        if self.tight_layout:
            plt.tight_layout()

        if not self.noshow:
            plt.show()

    def reset(self):
        self._taken_panel_nums = []
        self._additional_plots = []

    @staticmethod
    def __pp_utils_group_ax_by_panel(axes):
        grouped_axes = dict()
        for ax in axes:
            grouped_axes.setdefault(ax.bbox.bounds, [])
            grouped_axes[ax.bbox.bounds].append(ax)

        assert all(map(lambda x: len(x) <= 2, grouped_axes.values()))

        return grouped_axes

    def _pp_move_legend_outside(self, fig, axes):

        grouped_axes = self.__pp_utils_group_ax_by_panel(axes)

        for _axes in grouped_axes.values():
            # Only need to merge when both have legend
            if len(_axes) == 1:
                _axes[0].legend(bbox_to_anchor = (1.05, 0.9))
            else:
                if all(map(lambda x: x.get_legend() is not None, _axes)):
                    _axes[0].legend(bbox_to_anchor = (1.05, 0.9), loc='upper left')
                    _axes[1].legend(bbox_to_anchor = (1.05, 0.0), loc='lower left')
                elif any(map(lambda x: x.get_legend() is not None, _axes)):
                    _ax = _axes[0] if _axes[0].get_legend() is not None else _axes[1]
                    _ax.legend(bbox_to_anchor = (1.05, 0.9))

        return

    def _pp_merge_legend_for_each_panel(self, fig, axes):

        grouped_axes = self.__pp_utils_group_ax_by_panel(axes)

        for _axes in grouped_axes.values():
            # Only need to merge when both have legend
            if len(_axes) > 1 and all(map(lambda x: x.get_legend() is not None, _axes)):
                leg1, leg2 = _axes[0].get_legend(), _axes[1].get_legend()

                artists = (leg1.get_lines() + leg1.get_patches()
                            + leg2.get_lines() + leg2.get_patches())
                labels = [l.get_label() for l in artists]

                _axes[0].legend(artists, labels)
                _axes[1].legend([], [])
        
        return
        

    def postprocess(self, fig, axes):
        """Add y axis and etc"""

        if self.merge_legend_for_each_panel:
            self._pp_merge_legend_for_each_panel(fig, axes)
        if self.move_legend_outside:
            self._pp_move_legend_outside(fig, axes)
        return

    @property
    def panel_num(self) -> int:
        ret = 1 + len(self._taken_panel_nums)
        ret += 1 if self.volume else 0
        return ret

    @property
    def panel_ratios(self):
        return self._panel_ratios

    @panel_ratios.setter
    def panel_ratios(self, val):
        self._panel_ratios = val

    @property
    def plotter_args(self):
        ret = {'returnfig': True,
               'volume_panel': self.volume_panel,
               'volume': self.volume,
               'main_panel': self.main_panel,
               'style': self.style,
               }

        for _key, _val in [
                ('addplot', self._additional_plots),
                ('figsize', self.figsize),
                ('figscale', self.figscale),
                ('figratio', self.figratio),
                ('panel_ratios', self.panel_ratios),
                ('title', self.figure_title),
        ]:
            if _val is not None:
                ret[_key] = _val
        return ret

    def get_new_panel_num(self):
        for panel in range(0, 33):
            if panel == self.main_panel or  panel in self._taken_panel_nums:
                continue
            if self.volume and panel == self.volume_panel:
                continue
            self._taken_panel_nums.append(panel)
            return panel
        raise ValueError('The number of panel is 32 at most')

    def plot(self, df: pd.DataFrame, **kwargs):
        if self.tick_col in df.columns:
            df = df.set_index(self.tick_col)

        fig, axes = mpf.plot(df, **kwargs, **self.plotter_args) 
        self.postprocess(fig, axes)
        
        # Reset additional_plots
        self.reset()
        return fig, axes

    def plot_basic(
            self,
            df: pd.DataFrame,
            type: _T_TYPE_LITERAL = 'candle',
            mav: Optional[tuple[int, ...]] = None,
            **kwargs
    ):
        args = {'type': type}
        if mav is not None:
            args['mav'] = mav

        return self.plot(df, **args, **kwargs)

    def add_indicator(self,
                      indicator: _BaseIndicator,
                      *args,
                      new_panel: bool = False,
                      **kwargs):

        if new_panel or indicator.need_new_panel_num:
            new_panel_num = self.get_new_panel_num()
        else:
            new_panel_num = None

        _plotter_args = self.plotter_args.copy()
        _plotter_args['new_panel_num'] = new_panel_num

        self._additional_plots += indicator.make_addplot(_plotter_args, *args, **kwargs)
        return self


class OHLCMultiFigurePlotter(OHLCMpfPlotter):
    """Collection of different plot types for OHLC-type data
    """
    def __init__(
            self,
            tick_col: str,  # the columnt used as index
            ncols: int = 1,
            nrows: int = 1,
            figscale_main: Optional[float] = 1.,
            figratio_main: Optional[tuple[float, float]] = (4, 3),
            figsize_main: Optional[tuple[float, float]] = None,
            padding_main: float = 0.4,
            tight_layout: bool = False,
            noshow: bool = False,
            main_panel: int = 0,
            volume_panel: int = 1,
            volume: bool = False,
            figscale: Optional[float] = None,
            figratio: Optional[tuple[float, float]] = None,
            figsize: Optional[tuple[float, float]] = None,
            style: Literal['binance', 'binancedark', 'blueskies', 'brasil',
                           'charles', 'checkers', 'classic', 'default', 'ibd',
                           'kenan', 'mike', 'nightclouds', 'sas',
                           'starsandstripes', 'tradingview', 'yahoo'] = "default",
            panel_ratios: Optional[tuple[float, float] | tuple[float, ...] | list[float]] = None,
            title: Optional[str] = None,
            # Finer control in postprocess
            move_legend_outside: bool = False,
            merge_legend_for_each_panel: bool = True
    ):

        super().__init__(
            tick_col, tight_layout, noshow,
            main_panel, volume_panel, volume,
            figscale, figratio, figsize, style, panel_ratios,
            title,
            move_legend_outside=move_legend_outside,
            merge_legend_for_each_panel=merge_legend_for_each_panel
        )

        self.ncols = ncols
        self.nrows = nrows
        # Useful for subfigures
        self.figscale_main = figscale_main
        self.figratio_main = figratio_main
        self.padding_main = padding_main
        if figsize_main is None:
            w, h = figratio_main
            self.figsize_main: tuple[float, float] = (
                figscale_main * ( w * ncols + padding_main * (ncols - 1) ),
                figscale_main * ( h * nrows + padding_main * (nrows - 1) )
            )
        else:
            self.figsize_main = figsize_main

        self.fig_main = plt.figure(figsize=self.figsize_main)
        self.subfigs = self.fig_main.subfigures(nrows, ncols)
        if ncols == 1 or nrows == 1:
            self.current_subifg = self.subfigs[0]
        else:
            self.current_subifg = self.subfigs[0][0]
        
    def select_subfig(self, subfig_idx: int):

        if self.ncols == 1 or self.nrows == 1:
            self.current_subifg = self.subfigs[subfig_idx]
        else:
            _idx_col = subfig_idx % self.ncols
            _idx_row = subfig_idx // self.ncols
            self.current_subifg = self.subfigs[_idx_row][_idx_col]

    def set_subfig_suptitle(self, title):
        self.current_subifg.suptitle(title)

    @property
    def plotter_args(self):
        return {**super().plotter_args, 'subfig': self.current_subifg}
