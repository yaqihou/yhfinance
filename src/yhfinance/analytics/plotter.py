
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
            title: Optional[str] = None
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

    # TODO
    def move_legend_outside(self):
        pass

    # TODO
    def postprocess(self):
        """Add y axis and etc"""
        pass

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
               'main_panel': self.main_panel}

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

        ret = mpf.plot(df, **kwargs, **self.plotter_args) 
        # Reset additional_plots
        self.reset()
        return ret

    def plot_basic(
            self,
            df: pd.DataFrame,
            type: _T_TYPE_LITERAL = 'candle',
            mav=(3,),
            **kwargs
    ):
        return self.plot(df, type=type, mav=mav, **kwargs)

    # TODO - add historical recession overlay
    def add_macd_panel(self, df):
        new_panel_num = self.get_new_panel_num()

        self._additional_plots += [
            mpf.make_addplot(df[Col.Ind.Momentum.MACD.EMA12.name], color='lime', panel=self.main_panel, label='MACD-EMA12'),
            mpf.make_addplot(df[Col.Ind.Momentum.MACD.EMA26.name], color='c', panel=self.main_panel, label='MACD-EMA26'),
            # 
            mpf.make_addplot(df[Col.Ind.Momentum.MACD.MACD.name], color='fuchsia', panel=new_panel_num, label='MACD', secondary_y=True),
            mpf.make_addplot(df[Col.Ind.Momentum.MACD.Signal.name], color='b', panel=new_panel_num, label='Signal', secondary_y=True),
            mpf.make_addplot(df[Col.Ind.Momentum.MACD.MACD.name] -  df[Col.Ind.Momentum.MACD.Signal.name],
                             color='dimgray', panel=new_panel_num, type='bar', width=0.7, secondary_y=False)
        ]
        return self

    def add_rsi_panel(self,
                      df,
                      rsi_type: Literal['wilder', 'ema', 'cutler'] = 'cutler',
                      rsi_n: int = 14,
                      threshold: tuple[float, float] = (30, 70)
                      ):

        rsi_col = {'wilder': Col.Ind.Momentum.RSIWilder,
                   'ema': Col.Ind.Momentum.RSIEma,
                   'cutler': Col.Ind.Momentum.RSICutler}[rsi_type]
        
        new_panel_num = self.get_new_panel_num()

        _df_threshold = pd.DataFrame.from_dict({
            'upper': [threshold[1]] * len(df),
            'lower': [threshold[0]] * len(df)
        })
        self._additional_plots += [
            mpf.make_addplot(df[rsi_col.RSI.name + f"_{rsi_n}"], type='line', color='r', label=f'{rsi_type.capitalize()} RSI ({rsi_n})',
                             panel=new_panel_num),
            mpf.make_addplot(_df_threshold['upper'], type='line', color='k', linestyle='--', panel=new_panel_num),
            mpf.make_addplot(_df_threshold['lower'], type='line', color='k', linestyle='--', panel=new_panel_num)
        ]


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
        return {**super().plotter_args, 'subfig': self.current_subifg}
