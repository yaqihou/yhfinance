
import numpy as np
from numpy.typing import ArrayLike
import pandas as pd
import scipy.stats as scipy_stats 

import mplfinance as mpf

import abc
from ..ohlc_cols import Col, ColName, _T_RSI
from ..ohlc_processor import OHLCInterProcessor
from ..ohlc_data import OHLCData

from ._indicators_mixin import *
from ._base import _BaseIndicator
from .basics import IndSMA, IndEMA
from .moving_average_band import IndTriangularMovingAverage


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

        histogram_rise[0] = histogram[0]
        for idx, val in enumerate(histogram[1:], 1):
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


class _IndRSI(_PeriodMixin, _BaseIndicator):

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


class IndSpearman(_HistogramColorMixin, _PeriodMixin, _BaseIndicator):
    """The Spearman study is a technical indicator used for evaluation of
    trend strength and turning point detection. This study calculates
    Spearman's rank correlation coefficient in order to reveal correlation
    between actual price changes and extremely strong trend. The Spearman
    indicator registers two sets of data: prices of each bar during the
    specified period in their chronological order and the same prices
    sorted ascendingly. Spearman's ratio found for these two sets is then
    multiplied by 100; value of +100 suggests strong correlation with
    uptrend and -100, with downtrend.

    Since the extreme values of -100 and +100 are rarely recorded, values
    of +80 and -80 might be used as overbought and oversold levels
    respectively. Plotting the Spearman indicator along with its short SMA
    (e.g., three day SMA on daily charts) might give an idea when to expect
    the turning points: look for the crossovers of the two plots. Useful
    information can also be obtained by analyzing action of the indicator
    in relation to zero level: crossing above the zero level might be an
    opportunity for the Long Entry while falling below zero might suggest
    the Short Exit.
    """

    def __init__(
            self,
            data: OHLCData,
            price_col: ColName = Col.Close,
            period: int = 7,
            smoothing_period: int = 10
    ):
        self.smoothing_period = smoothing_period
        super().__init__(data, price_col, period=period)

    def _calc(self) -> pd.DataFrame:

        _df = self._df[[self.tick_col]].copy()

        prices = self.df[self.price_col].values
        spearman = np.full(len(_df), np.nan)

        for idx in range(self.period, len(_df)+1):
            # TODO - this is very inefficient. Could optimize
            stat_res = scipy_stats.spearmanr(
                prices[idx-self.period:idx],
                np.sort(prices[idx-self.period:idx])
            )
            spearman[idx-1] = stat_res.statistic

        _df[Col.Ind.Spearman.SpearmanRCC(self.period)] = spearman
        _df[Col.Ind.Spearman.SpearmanSMA(self.period, self.smoothing_period)] = (
            _df[Col.Ind.Spearman.SpearmanRCC(self.period)].rolling(self.smoothing_period)\
                                                          .mean()
        )

        return _df

    @property
    def values(self):
        return self._df[Col.Ind.Spearman.SpearmanRCC(self.period)].values
    
    @property
    def need_new_panel_num(self) -> bool:
        return True

    def make_addplot(self, plotter_args: dict,
                     *args,
                     **kwargs) -> list[dict]:

        histogram = (self.df[Col.Ind.Spearman.SpearmanRCC(self.period)]
                     - self.df[Col.Ind.Spearman.SpearmanSMA(self.period, self.smoothing_period)]).values
        histogram_color = self._get_histogram_color(histogram)
        
        new_panel_num = plotter_args['new_panel_num']

        return [
            mpf.make_addplot(
                self.df[Col.Ind.Spearman.SpearmanRCC(self.period)],
                panel=new_panel_num,
                color='fuchsia',
                secondary_y=True,
                label=Col.Ind.Spearman.SpearmanRCC(self.period)
            ),
            mpf.make_addplot(
                self.df[Col.Ind.Spearman.SpearmanSMA(self.period, self.smoothing_period)],
                panel=new_panel_num,
                color='b',
                secondary_y=True,
                label=Col.Ind.Spearman.SpearmanSMA(self.period, self.smoothing_period)
            ),
            # Histogram
            mpf.make_addplot(
                histogram,
                panel=new_panel_num,
                color=histogram_color,
                type='bar', width=0.7, secondary_y=False),
        ]


class IndRelativeVigorIndex(_HistogramColorMixin, _PeriodMixin, _BaseIndicator):
    """The Relative Vigor Index (RVI) is a momentum indicator used in
    technical analysis. It measures the strength of a trend by comparing a
    security's closing price to its trading range while smoothing the
    results using a simple moving average (SMA).

    The RVI's usefulness is based on the observed tendency for prices to
    close higher than they open during uptrends, and to close lower than
    they open in downtrends."""

    def __init__(
            self,
            data: OHLCData,
            price_col: ColName = Col.Close,
            period: int = 7,
            smoothing_weight: tuple[int | float, ...] | list[int | float] = (1, 2, 2, 1)
    ):
        self.smoothing_weight = smoothing_weight
        super().__init__(data, price_col, period=period)

    def _calc(self) -> pd.DataFrame:

        _df = self._df[[self.tick_col]].copy()

        C = self.df[Col.Close].values
        O = self.df[Col.Open].values
        L = self.df[Col.Low].values
        H = self.df[Col.High].values

        # TODO - use volume as a weight
        # TODO - now only the ratio is considered, but the magnitude of swing is out of the picture

        # NOTE - the convolution is easy to implement but does not support a dynamic Volume weight
        nume = np.convolve(C - O, self.smoothing_weight, mode='full')[:len(_df)]
        deno = np.convolve(H - L, self.smoothing_weight, mode='full')[:len(_df)]
        
        # nume_list = [C - O]
        # for padding in range(1, len(self.smoothing_weight)):
        #     nume_list.append(
        #         np.pad(C[:-padding] - O[:-padding], (padding, 0), 'constant', constant_values=(np.nan,))
        #     )

        # deno_list = [H - L]
        # for padding in range(1, len(self.smoothing_weight)):
        #     deno_list.append(
        #         np.pad(H[:-padding] - L[:-padding], (padding, 0), 'constant', constant_values=(np.nan,))
        #     )

        # nume_arr = np.array(nume_list)
        # deno_arr = np.array(deno_list)

        # nume = (nume_arr * np.array(self.smoothing_weight)[:, None]).sum(axis=0) / sum(self.smoothing_weight)
        # deno = (deno_arr * np.array(self.smoothing_weight)[:, None]).sum(axis=0) / sum(self.smoothing_weight)

        # a = C - O
        # b = np.pad(C[:-1] - O[:-1], (1, 0), 'constant', constant_values=(np.nan,))
        # c = np.pad(C[:-2] - O[:-2], (2, 0), 'constant', constant_values=(np.nan,))
        # d = np.pad(C[:-3] - O[:-3], (3, 0), 'constant', constant_values=(np.nan,))

        # e = H - L
        # f = np.pad(H[:-1] - L[:-1], (1, 0), 'constant', constant_values=(np.nan,))
        # g = np.pad(H[:-2] - L[:-2], (2, 0), 'constant', constant_values=(np.nan,))
        # h = np.pad(H[:-3] - L[:-3], (3, 0), 'constant', constant_values=(np.nan,))

        # nume_test = ( a + 2*b + 2*c + d) / 6.
        # deno_test = ( e + 2*f + 2*g + h) / 6.
        # print(np.nansum((nume_test - nume)))
        # print(np.nansum((deno_test - deno)))

        _df[Col.Ind.RVigorI.Nume.name] = nume
        _df[Col.Ind.RVigorI.Deno.name] = deno

        _df[Col.Ind.RVigorI.RVI(self.period)] = (
            _df[Col.Ind.RVigorI.Nume.name].rolling(self.period).mean()
            / _df[Col.Ind.RVigorI.Deno.name].rolling(self.period).mean()
        )

        rvi = _df[Col.Ind.RVigorI.RVI(self.period)].values
        rvi_list = [rvi]
        for padding in range(1, len(self.smoothing_weight)):
            rvi_list.append(
                np.pad(rvi[:-padding], (padding, 0), 'constant', constant_values=(np.nan,))
            )
        rvi_arr = np.array(rvi_list)
        signal = (rvi_arr * np.array(self.smoothing_weight)[:, None]).sum(axis=0) / sum(self.smoothing_weight)

        # rvi = _df[Col.Ind.RVigorI.RVI(self.period)].values
        # i = np.pad(rvi[:-1], (1, 0), 'constant', constant_values=(np.nan,))
        # j = np.pad(rvi[:-2], (2, 0), 'constant', constant_values=(np.nan,))
        # k = np.pad(rvi[:-3], (3, 0), 'constant', constant_values=(np.nan,))
        # signal_test = (rvi + 2*i + 2*j + k) / 6.
        # print(np.nansum((signal_test - signal)))

        _df[Col.Ind.RVigorI.Signal(self.period)] = signal

        return _df

    @property
    def values(self):
        return self._df[Col.Ind.RVigorI.RVI(self.period)].values
    
    @property
    def need_new_panel_num(self) -> bool:
        return True

    def make_addplot(self, plotter_args: dict,
                     *args,
                     **kwargs) -> list[dict]:

        histogram = (self.df[Col.Ind.RVigorI.RVI(self.period)]
                     - self.df[Col.Ind.RVigorI.Signal(self.period)]).values
        histogram_color = self._get_histogram_color(histogram)
        
        new_panel_num = plotter_args['new_panel_num']

        return [
            mpf.make_addplot(
                self.df[Col.Ind.RVigorI.RVI(self.period)],
                panel=new_panel_num,
                color='fuchsia',
                secondary_y=True,
                label=Col.Ind.RVigorI.RVI(self.period)
            ),
            mpf.make_addplot(
                self.df[Col.Ind.RVigorI.Signal(self.period)],
                panel=new_panel_num,
                color='b',
                secondary_y=True,
                label=Col.Ind.RVigorI.Signal(self.period)
            ),
            # Histogram
            mpf.make_addplot(
                histogram,
                panel=new_panel_num,
                color=histogram_color,
                type='bar', width=0.7, secondary_y=False),
        ]


class IndRelativeVigorIndexTMA(_HistogramColorMixin, _PeriodMixin, _BaseIndicator):
    """A variation of the ordinary RVI in that the smoothing is done through TMA
    """

    def __init__(
            self,
            data: OHLCData,
            price_col: ColName = Col.Close,
            period: int = 7,
            period_signal: int = 4
    ):
        self.period_signal = period_signal
        super().__init__(data, price_col, period=period)

    def _calc(self) -> pd.DataFrame:

        _df = self._df[[self.tick_col]].copy()

        C = self.df[Col.Close].values
        O = self.df[Col.Open].values
        L = self.df[Col.Low].values
        H = self.df[Col.High].values

        ratios = (C - O) / (H - L)
        _df[Col.Ind.RVigorI_TMA.RV.name] = ratios

        _tma = IndTriangularMovingAverage(
            OHLCData(_df, self.tick_col),
            price_col=Col.Ind.RVigorI_TMA.RV.name,
            period=self.period
        )
        _df[Col.Ind.RVigorI_TMA.RVI(self.period)] = _tma.values


        _tma = IndTriangularMovingAverage(
            OHLCData(_df, self.tick_col),
            price_col=Col.Ind.RVigorI_TMA.RVI(self.period),
            period=self.period_signal
        )
        _df[Col.Ind.RVigorI_TMA.Signal(self.period, self.period_signal)] = _tma.values

        return _df

    @property
    def values(self):
        return self._df[Col.Ind.RVigorI.RVI(self.period)].values
    
    @property
    def need_new_panel_num(self) -> bool:
        return True

    def make_addplot(self, plotter_args: dict,
                     *args,
                     **kwargs) -> list[dict]:

        histogram = (self.df[Col.Ind.RVigorI_TMA.RVI(self.period)]
                     - self.df[Col.Ind.RVigorI_TMA.Signal(self.period, self.period_signal)]).values
        histogram_color = self._get_histogram_color(histogram)
        
        new_panel_num = plotter_args['new_panel_num']

        return [
            mpf.make_addplot(
                self.df[Col.Ind.RVigorI_TMA.RVI(self.period)],
                panel=new_panel_num,
                color='fuchsia',
                secondary_y=True,
                label=Col.Ind.RVigorI_TMA.RVI(self.period)
            ),
            mpf.make_addplot(
                self.df[Col.Ind.RVigorI_TMA.Signal(self.period, self.period_signal)],
                panel=new_panel_num,
                color='b',
                secondary_y=True,
                label=Col.Ind.RVigorI_TMA.Signal(self.period, self.period_signal)
            ),
            # Histogram
            mpf.make_addplot(
                histogram,
                panel=new_panel_num,
                color=histogram_color,
                type='bar', width=0.7, secondary_y=False),
        ]
