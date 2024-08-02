# 

import numpy as np
from numpy.typing import ArrayLike
import pandas as pd

import mplfinance as mpf

import abc
from ..ohlc_cols import Col, ColName
from ..ohlc_data import OHLCData

from ._indicators_mixin import *
from ._base import _BaseIndicator


class IndSMA(_RollingMixin, _BaseIndicator):

    def __init__(
            self,
            data      : OHLCData,
            period    : int = 7,
            price_col : ColName = Col.Close,  # if None, the same as multiplier
    ):

        super().__init__(data, period=period, price_col=price_col)

    def _calc(self) -> pd.DataFrame:

        _df = self._df[[self.tick_col]].copy()
        _df[Col.Ind.SMA(self.period)] = self._df[self.price_col.name].rolling(self.period).mean()

        return _df

    @property
    def values(self):
        return self._df[Col.Ind.SMA(self.period)].values

    def make_addplot(self, plotter_args: dict, *args, **kwargs) -> list[dict]:
        return [
            mpf.make_addplot(
                self._df[Col.Ind.SMA(self.period)],
                type='line', panel=plotter_args['main_panel'],
                label=Col.Ind.SMA(self.period)
            )
        ]


class IndEMA(_RollingMixin, _BaseIndicator):
    """Return the exponential moving average"""

    def __init__(
            self,
            data      : OHLCData,
            period    : int = 7,
            price_col : ColName = Col.Close,  # if None, the same as multiplier
    ):

        super().__init__(data, period=period, price_col=price_col)

    def _calc(self) -> pd.DataFrame:

        _df = self._df[[self.tick_col]].copy()
        _df[Col.Ind.EMA(self.period)] = self.df[self.price_col.name].ewm(
            span=self.period, adjust=False).mean()

        return _df

    @property
    def values(self):
        return self._df[Col.Ind.EMA(self.period)].values

    def make_addplot(self, plotter_args: dict, *args, **kwargs) -> list[dict]:
        return [
            mpf.make_addplot(
                self.df[Col.Ind.EMA(self.period)],
                type='line', panel=plotter_args['main_panel'],
                label=Col.Ind.EMA(self.period)
            )
        ]


class IndSMMA(_RollingMixin, _BaseIndicator):
    """Return the exponential moving average"""

    def __init__(
            self,
            data      : OHLCData,
            period    : int = 7,
            price_col : ColName = Col.Close,  # if None, the same as multiplier
    ):

        super().__init__(data, period=period, price_col=price_col)

    def _calc(self) -> pd.DataFrame:

        _df = self._df[[self.tick_col]].copy()

        vals = self.df[self.price_col.name].copy()
        vals.iloc[self.period] = vals.iloc[:self.period].mean()
        vals.iloc[:self.period] = pd.NA
        
        alpha = 1 / self.period
        _df[Col.Ind.SMMA(self.period)] = vals.ewm(alpha=alpha, ignore_na=True, adjust=False).mean()

        return _df

    @property
    def values(self):
        return self._df[Col.Ind.SMMA(self.period)].values

    def make_addplot(self, plotter_args: dict, *args, **kwargs) -> list[dict]:
        return [
            mpf.make_addplot(
                self.df[Col.Ind.SMMA(self.period)],
                type='line', panel=plotter_args['main_panel'],
                label=Col.Ind.EMA(self.period)
            )
        ]
