
import numpy as np
from numpy.typing import ArrayLike
import pandas as pd

import mplfinance as mpf

import abc
from ..ohlc_cols import Col, ColName, _T_RSI
from ..ohlc_processor import OHLCInterProcessor
from ..ohlc_data import OHLCData

from ._indicators_mixin import *
from ._base import _BaseIndicator
from .basics import IndSMA, IndEMA


# class _IndMACDLike(_baseIndicator):
#     """For indicators which the values are taken from two trend lines to
#     get the value (MACD / AO), and then a signal line is created from it or
#     provided.

#     Hence they could share the similar function to draw the plot
#     """

#     pass


class IndAwesomeOscillator(_BaseIndicator):
    """The Awesome Oscillator is an indicator used to measure market
    momentum. AO calculates the difference of a 34 Period and 5 Period
    Simple Moving Averages. The Simple Moving Averages that are used are
    not calculated using closing price but rather each bar's midpoints. AO
    is generally used to affirm trends or to anticipate possible reversals.
    """
    

    def __init__(
            self,
            data          : OHLCData,
            period_fast   : int           = 5,
            period_slow   : int           = 34,
            price_col     : ColName       = Col.Close,
    ):
        """
        period: for SMA, usually between 5 and 10
        multiplier: for ATR scale, commonly take to be 2
        """
        self.period_fast = period_fast
        self.period_slow = period_slow
        super().__init__(data, price_col=price_col)

    def _calc(self) -> pd.DataFrame:

        self._sma_fast = IndSMA(self._data, self.period_fast, price_col=self.price_col)
        _sma_fast_val = self._sma_fast.df[Col.Ind.SMA(self.period_fast)].values

        self._sma_slow = IndSMA(self._data, self.period_slow, price_col=self.price_col)
        _sma_slow_val = self._sma_slow.df[Col.Ind.SMA(self.period_slow)].values

        _df = self._df[[self.tick_col]].copy()
        _df[Col.Ind.AwesomeOscillator.Fast(self.period_fast)] = _sma_fast_val
        _df[Col.Ind.AwesomeOscillator.Slow(self.period_slow)] = _sma_slow_val
        _df[Col.Ind.AwesomeOscillator.AO(self.period_fast, self.period_slow)] = \
            _sma_fast_val - _sma_slow_val

        return _df

    @property
    def values(self):
        return self.df[Col.Ind.AwesomeOscillator.AO(self.period_fast, self.period_slow)].values

    @property
    def need_new_panel_num(self) -> bool:
        return True

    def make_addplot(self, plotter_args: dict,
                     *args,
                     with_sma: bool = False,
                     **kwargs) -> list[dict]:
        kwargs = {
            'label': Col.Ind.AwesomeOscillator.AO(self.period_fast, self.period_slow),
            'secondary_y': False,
            'panel': plotter_args['new_panel_num']
        }

        ret = [mpf.make_addplot(
            self.df[Col.Ind.AwesomeOscillator.AO(self.period_fast, self.period_slow)],
            **kwargs)]

        if with_sma:
            # Ensure we have the same size
            self._sma_fast._df = self._sma_fast._df.merge(
                self.df[[self.tick_col]], how='inner', on=self.tick_col)
            self._sma_slow._df = self._sma_fast._df.merge(
                self.df[[self.tick_col]], how='inner', on=self.tick_col)
            ret += self._sma_fast.make_addplot(plotter_args)
            ret += self._sma_slow.make_addplot(plotter_args)
        
        return ret


# class IndADOscillator(_BaseIndicator):
#     """Acceleration/Deceleration (AC) is a histogram-type oscillator that
#     quantifies the current market momentum. The longer the bars are, the
#     greater the market momentum's acceleration/deceleration, and vice
#     versa. The AC values are calculated as the difference between Awesome
#     Oscillator (AO) and its 5-period Simple Moving Average (SMA).
#     """

#     def __init__(
#             self,
#             data          : OHLCData,
#             period_signla
#             period_fast   : int           = 5,
#             period_slow   : int           = 34,
#             price_col     : ColName       = Col.Close,
#     ):
#         self.period_fast = period_fast 


class IndMACD(_BaseIndicator):

    def __init__(
            self,
            data              : OHLCData,
            short_term_window : int     = 12,
            long_term_window  : int     = 26,
            signal_window     : int     = 9,
            price_col         : ColName = Col.Close
    ):

        self.short_term_window = short_term_window
        self.long_term_window  = long_term_window
        self.signal_window     = signal_window

        super().__init__(data, price_col=price_col)

    def _calc(self) -> pd.DataFrame:
        
        _ewm_short = IndEMA(self._data, period=self.short_term_window, price_col=self.price_col)
        ewm_short = _ewm_short.df[Col.Ind.EMA(self.short_term_window)]

        _ewm_long = IndEMA(self._data, period=self.long_term_window, price_col=self.price_col)
        ewm_long = _ewm_long.df[Col.Ind.EMA(self.long_term_window)]

        macd = ewm_short - ewm_long
        signal = macd.ewm(span=self.signal_window, adjust=False).mean()

        df = self._df[[self.tick_col]].copy()
        
        df[Col.Ind.MACD.EMA12.name] = ewm_short
        df[Col.Ind.MACD.EMA26.name] = ewm_long
        df[Col.Ind.MACD.MACD(
            self.short_term_window, self.long_term_window, self.signal_window
        )] = macd

        df[Col.Ind.MACD.Signal(
            self.short_term_window, self.long_term_window, self.signal_window)] = signal

        return df

    # TODO - could add a argument to control this in init
    @property
    def need_new_panel_num(self) -> bool:
        return True

    def make_addplot(self, plotter_args: dict, *args, **kwargs) -> list[dict]:

        main_panel = plotter_args['main_panel']
        new_panel_num = plotter_args['new_panel_num']

        _col_macd = Col.Ind.MACD.MACD(
            self.short_term_window, self.long_term_window, self.signal_window)
        _col_signal = Col.Ind.MACD.Signal(
            self.short_term_window, self.long_term_window, self.signal_window)

        # Add color to histogram
        histogram = (self.df[_col_macd] -  self.df[_col_signal]).values
        # Rising trend
        histogram_rise = np.full_like(histogram, np.nan)
        # Decreasing trend 
        histogram_fall = np.full_like(histogram, np.nan)

        for idx, val in enumerate(histogram[1:]):
            if val > histogram[idx-1]:
                histogram_rise[idx] = histogram[idx]
            else:
                histogram_fall[idx] = histogram[idx]

        return [
            mpf.make_addplot(
                self.df[Col.Ind.MACD.EMA12.name], linestyle='--',
                panel=main_panel, label='MACD-EMA12'),
            mpf.make_addplot(
                self.df[Col.Ind.MACD.EMA26.name], linestyle=':',
                panel=main_panel, label='MACD-EMA26'),
            # 
            mpf.make_addplot(self.df[_col_macd],
                             color='fuchsia', panel=new_panel_num,
                             label='MACD', secondary_y=True),
            mpf.make_addplot(self.df[_col_signal],
                             color='b', panel=new_panel_num, label='Signal', secondary_y=True),
            # TODO - use style value from mplfinance
            # TODO - check how does mpf implement the volume
            mpf.make_addplot(histogram_rise,
                             color='lime', panel=new_panel_num,
                             type='bar', width=0.7, secondary_y=False),
            mpf.make_addplot(histogram_fall,
                             color='r', panel=new_panel_num,
                             type='bar', width=0.7, secondary_y=False)
        ]

    @property
    def values(self):
        return {
            'Short': self.df[Col.Ind.MACD.EMA12].values,
            'Long': self.df[Col.Ind.MACD.EMA26].values,
            'MACD': self.df[
                Col.Ind.MACD.MACD(
                    self.short_term_window, self.long_term_window, self.signal_window)
            ].values,
            'Signal': self.df[
                Col.Ind.MACD.Signal(
                    self.short_term_window, self.long_term_window, self.signal_window)
            ].values,
        }


class _IndRSI(_RollingMixin, _BaseIndicator):

    def __init__(
            self,
            data      : OHLCData,
            period    : int                 = 14,
            price_col : ColName | str       = Col.Close,
            threshold : tuple[float, float] = (30, 70)
    ):
        self.ewm_ups: pd.DataFrame
        self.ewm_dns: pd.DataFrame
        self.threshold = threshold
        super().__init__(data, period=period, price_col=price_col)

    @property
    def need_new_panel_num(self) -> bool:
        return True

    def _get_gl_for_rsi(self) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:

        _inter_processor = OHLCInterProcessor(self._data, tick_offset=-1)
        _inter_processor.add_all_return()
        _col_res = Col.Inter.get_return_col(self.price_col)
            
        _df = _inter_processor.get_result()
        _df = _df[[self.tick_col, _col_res.gl]].copy()

        ups = _df[_col_res.gl].apply(lambda x: max(x, 0))
        dns = _df[_col_res.gl].apply(lambda x: -min(x, 0))
        _df.drop(columns=[_col_res.gl], inplace=True)

        return _df, ups, dns

    @property
    @abc.abstractmethod
    def rsi_col(self) -> _T_RSI:
        """Return the col used for this type of RSI"""

    @property
    def rsi_type(self) -> str:
        return self.rsi_col.RSI.name

    @property
    def values(self):
        return self.df[self.rsi_col.RSI(self.period)].values

    def _assign_rsi_result(self, df) -> pd.DataFrame:
        
        df[self.rsi_col.AvgGain(self.period)] = self.ewm_ups
        df[self.rsi_col.AvgLoss(self.period)] = self.ewm_dns
        df[self.rsi_col.RS(self.period)] = self.ewm_ups / self.ewm_dns
        df[self.rsi_col.RSI(self.period)] = 100 - (100 / (1 + self.ewm_ups / self.ewm_dns))

        return df

    def make_addplot(self, plotter_args: dict, *args, **kwargs) -> list[dict]:

        new_panel_num = plotter_args['new_panel_num']

        _df_threshold = pd.DataFrame.from_dict({
            'upper': [max(self.threshold)] * len(self.df),
            'lower': [min(self.threshold)] * len(self.df)
        })
        
        return [
            mpf.make_addplot(self.df[self.rsi_col.RSI(self.period)],
                             type='line', color='r',
                             label=self.rsi_col.RSI(self.period),
                             panel=new_panel_num, secondary_y=False),
            #
            mpf.make_addplot(_df_threshold['upper'], type='line', color='k',
                             linestyle='--', panel=new_panel_num, secondary_y=False),
            #
            mpf.make_addplot(_df_threshold['lower'], type='line', color='k',
                             linestyle='--', panel=new_panel_num, secondary_y=False)
        ]


class IndWilderRSI(_IndRSI):
    """Using the original formula from Wilder U / D is the difference
        between close prices, using SMMA with alpha = 1 / n, i.e. y_t =
        (1-a) * y_t-1 + a * x_t First n -1 data is set to nan, and initial
        result starts from nth element as the
    """

    @property
    def rsi_col(self) -> _T_RSI:
        return Col.Ind.RSIWilder

    def _calc(self) -> pd.DataFrame:
        _df, ups, dns = self._get_gl_for_rsi()
        n = self.period

        ups.iloc[n] = ups.iloc[:n].mean()
        ups.iloc[:n] = pd.NA
        dns.iloc[n] = dns.iloc[:n].mean()
        dns.iloc[:n] = pd.NA

        alpha = 1 / n
        self.ewm_ups = ups.ewm(alpha=alpha, ignore_na=True, adjust=False).mean()
        self.ewm_dns = dns.ewm(alpha=alpha, ignore_na=True, adjust=False).mean()

        return self._assign_rsi_result(_df)


class IndEmaRSI(_IndRSI):
    """The main difference from wilder's original is that instead of SMMA,
    we used a EMA so that the first n obs are NOT nan"""

    @property
    def rsi_col(self) -> _T_RSI:
        return Col.Ind.RSIEma

    def _calc(self) -> pd.DataFrame:
        _df, ups, dns = self._get_gl_for_rsi()

        alpha = 1 / self.period
        self.ewm_ups = ups.ewm(alpha=alpha, adjust=False).mean()
        self.ewm_dns = dns.ewm(alpha=alpha, adjust=False).mean()

        return self._assign_rsi_result(_df)


class IndCutlerRSI(_IndRSI):
    """Cutler's RSI variation is based on SMA, to overcome the so-called 'Data Length Dependency'"""

    @property
    def rsi_col(self) -> _T_RSI:
        return Col.Ind.RSICutler

    def _calc(self) -> pd.DataFrame:
        _df, ups, dns = self._get_gl_for_rsi()

        self.ewm_ups = ups.rolling(self.period).mean()
        self.ewm_dns = dns.rolling(self.period).mean()

        return self._assign_rsi_result(_df)

