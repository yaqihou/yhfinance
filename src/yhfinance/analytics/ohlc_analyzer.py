

from collections import namedtuple

import pandas as pd
import datetime as dt

from yhfinance.analytics.const import ColName

from ._base import _OHLCBase
from .const import Col, ColName

_SimpleGlRtn = namedtuple(
    "SimpleGlRtn",
    ['gl', 'rtn', 'log_rtn',
     'buy_at', 'sell_at',
     'buy_tick', 'sell_tick'])


class OHLCSimpleStrategist(_OHLCBase):
    """Applying a series of strategy on the input df.
    """

    def __init__(self, df: pd.DataFrame, tick_col: str = Col.Date.name):
        super().__init__(df, tick_col)

        _df = self._df[[self.tick_col] + list(Col.OHLC)].set_index(self.tick_col)

        self._df_rolling_min = _df\
                               .rolling(len(self._df), min_periods=1).min()\
                               .reset_index()\
                               .rename(columns={
                                   x: x+"RollingMin" for x in Col.OHLC
                               })
        self._df_rolling_max = _df\
                               .rolling(len(self._df), min_periods=1).max()\
                               .reset_index()\
                               .rename(columns={
                                   x: x+"RollingMax" for x in Col.OHLC
                               })

        self._df_rolling = self._df_rolling_min.merge(
            self._df_rolling_max, on=self.tick_col
        )
        self._df_rolling = self._df_rolling.merge(_df.reset_index(), on=self.tick_col)

    def _get_gl_rtn(self, buy_at: str | float, sell_at: str | float):

        _df = pd.DataFrame(columns=['Buy', 'Sell', 'Gl', 'Rtn'])
        if isinstance(buy_at, (int, float)):
            _df['Buy'] = [buy_at] * len(self._df)
        else:
            _df['Buy'] = self._df_rolling[buy_at]

        if isinstance(sell_at, (int, float)):
            _df['Sell'] = [sell_at] * len(self._df)
        else:
            _df['Sell'] = self._df_rolling[sell_at]

        _df['Gl'] = _df['Sell'] - _df['Buy']
        _df['Rtn'] = _df['Gl'] / _df['Rtn']

        return _df

    def buy_at_first_day_gl(self, buy_at: ColName = Col.Open):
        """Return the gain/loss if buy at the first day"""

        _df = self._get_gl_rtn(
            buy_at=self._df.iloc[0, :].loc[:, buy_at.name],
            sell_at=Col.High.name)

    def max_gain(self, buy_at: ColName = Col.Low,  # Could also be Close / Open
                 sell_at: ColName = Col.High,
                 allow_intraday_trading: bool = False,  # TODO - if buy and high could happen at the same day
                 return_raw: bool = False
                 ):
        """Return the max possible gain during this period, note that max
        gain could happen differently from the max return"""

        _df = self._get_gl_rtn(buy_at=Col.Low.name + 'RollingMin', sell_at=Col.High.name)
        if return_raw:
            return _df
        raise NotImplementedError("TODO")

    def max_loss(self, buy_at: ColName = Col.High,
                 sell_at: ColName = Col.Low,
                 allow_intraday_trading: bool = False,  # TODO - if buy and high could happen at the same day
                 return_raw: bool = False
                 ):
        """Return the max possible loss during this period"""
        _df = self._get_gl_rtn(buy_at=Col.Low.name + 'RollingMax', sell_at=Col.Low.name)
        if return_raw:
            return _df
        raise NotImplementedError("TODO")


class OHLCStreakAnalyzer(_OHLCBase):

    # TODO - add sanity check to make sure the columns have streak to analysis

    pass
