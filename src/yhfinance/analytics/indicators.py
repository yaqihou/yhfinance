# 
import heapq
from typing import Optional

import numpy as np
import pandas as pd

import mplfinance as mpf

import abc
from .const import Col, ColName, _T_RSI
from .ohlc_processor import OHLCInterProcessor
from .defs import OHLCData, OHLCDataBase

from ._indicators_mixin import *


__all__ = [
    'IndSMA',
    'IndMACD',
    'IndWilderRSI', 'IndEmaRSI', 'IndCutlerRSI',
    'IndTrueRange', 'IndAvgTrueRange', 'IndATRBand', 'IndStarcBand',
    'IndSupertrend',
    'IndAroon',
    'IndAwesomeOscillator']

# TODO - add plotter configs into each class so that could be used outside
# TODO - add related indicators as class property
# TODO - add SMMA, EMA and other base indicators


class _BaseIndicator(OHLCDataBase, abc.ABC):

    # TODO - need to implement for each indicator
    _category: str = "undefined"
    _abbrev: str = 'undefined'

    def __init__(self, data: OHLCData):
        super().__init__(data)

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

    # TODO - return the final results as array / list of array
    # @property
    # def values(self) -> np.ndarray:
    #     return
    

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


# -----------------------------------
# Implementation starts below

class IndSMA(_RollingMixin, _PricePickMixin, _BaseIndicator):

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

    def make_addplot(self, plotter_args: dict, *args, **kwargs) -> list[dict]:
        return [
            mpf.make_addplot(
                self._df[Col.Ind.SMA(self.period)],
                type='line', panel=plotter_args['main_panel'],
                label=Col.Ind.SMA(self.period)
            )
        ]

class IndEMA(_RollingMixin, _PricePickMixin, _BaseIndicator):
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

    def make_addplot(self, plotter_args: dict, *args, **kwargs) -> list[dict]:
        return [
            mpf.make_addplot(
                self.df[Col.Ind.EMA(self.period)],
                type='line', panel=plotter_args['main_panel'],
                label=Col.Ind.EMA(self.period)
            )
        ]


class IndSMMA(_RollingMixin, _PricePickMixin, _BaseIndicator):
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

    def make_addplot(self, plotter_args: dict, *args, **kwargs) -> list[dict]:
        return [
            mpf.make_addplot(
                self.df[Col.Ind.SMMA(self.period)],
                type='line', panel=plotter_args['main_panel'],
                label=Col.Ind.EMA(self.period)
            )
        ]

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
        self.price_col         = price_col

        super().__init__(data)

    def _calc(self) -> pd.DataFrame:
        
        _ewm_short = IndEMA(self._data, period=self.short_term_window)
        ewm_short = _ewm_short.df[Col.Ind.EMA(self.short_term_window)]

        _ewm_long = IndEMA(self._data, period=self.long_term_window)
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


class _IndRSI(_RollingMixin, _BaseIndicator):

    def __init__(
            self,
            data      : OHLCData,
            period    : int                 = 14,
            threshold : tuple[float, float] = (30, 70)
    ):
        self.ewm_ups: pd.DataFrame
        self.ewm_dns: pd.DataFrame
        self.threshold = threshold
        super().__init__(data, period=period)

    @property
    def need_new_panel_num(self) -> bool:
        return True

    def _get_gl_for_rsi(self) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:

        _df = OHLCInterProcessor(self._data, tick_offset=-1).add_close_return().get_result()
        _df = _df[[self.tick_col, Col.Inter.CloseReturn.gl]].copy()

        ups = _df[Col.Inter.CloseReturn.gl].apply(lambda x: max(x, 0))
        dns = _df[Col.Inter.CloseReturn.gl].apply(lambda x: -min(x, 0))
        _df = _df.drop(columns=[Col.Inter.CloseReturn.gl])

        return _df, ups, dns

    @property
    @abc.abstractmethod
    def rsi_col(self) -> _T_RSI:
        """Return the col used for this type of RSI"""

    @property
    def rsi_type(self) -> str:
        return self.rsi_col.RSI.name

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

    def _calc(self) -> None:
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

    def _calc(self) -> None:
        _df, ups, dns = self._get_gl_for_rsi()

        self.ewm_ups = ups.rolling(self.period).mean()
        self.ewm_dns = dns.rolling(self.period).mean()

        return self._assign_rsi_result(_df)


class IndTrueRange(_RollingMixin, _BaseIndicator):

    def __init__(
            self,
            data   : OHLCData,
            period : int = 14
    ):
    
        super().__init__(data, period=period)

    def _calc(self) -> pd.DataFrame:
        _df = OHLCInterProcessor(self._data, tick_offset=-1)._df_offset

        # TR = max[(H-L), abs(H-Cp), abs(L-Cp)]
        _df[Col.Ind.TrueRange.name] = pd.concat([
            _df[Col.High.cur] - _df[Col.Low.cur],
            (_df[Col.High.cur] - _df[Col.Close.sft]).abs(),
            (_df[Col.Low.cur] - _df[Col.Close.sft]).abs()
            ], axis=1).max(axis=1)

        return _df

    def make_addplot(self, plotter_args: dict, *args, **kwargs) -> list[dict]:
        raise NotImplementedError()
    

class IndAvgTrueRange(_RollingMixin, _BaseIndicator):

    def __init__(
            self,
            data: OHLCData,
            period   : int = 14,
            keep_tr_result: bool = True
    ):
    
        self.keep_tr_result = keep_tr_result
        super().__init__(data, period=period)

    def _calc(self) -> pd.DataFrame:
        # ATR_curr = ATR_prev * (n-1) / n + TR_curr * (1/n)
        # i.e. a EWM with alpha = 1/n
        # initial condition is ATR_0 = mean(sum(TR_n))
        period = self.period

        ind_tr = IndTrueRange(self._data, period)
        _df = ind_tr.get_result()

        _tr = _df[Col.Ind.TrueRange.name].copy()
        _tr.iloc[period] = _tr.iloc[:period].mean()
        _tr.iloc[:period] = pd.NA

        alpha = 1 / period
        _df[Col.Ind.AvgTrueRange(period)] = _tr.ewm(alpha=alpha, ignore_na=True, adjust=False).mean()

        if not self.keep_tr_result:
            _df = _df[[self.tick_col, Col.Ind.AvgTrueRange(period)]]

        return _df

    @property
    def need_new_panel_num(self) -> bool:
        return True

    def make_addplot(self, plotter_args: dict, *args, **kwargs) -> list[dict]:
        _ = plotter_args
        _ = args
        _ = kwargs

        return [
            mpf.make_addplot(
                self.df[Col.Ind.AvgTrueRange(self.period)],
                type='line',
                panel=plotter_args['new_panel_num'],
                label=Col.Ind.AvgTrueRange(self.period)
            )
        ]


class IndATRBand(_BandMixin, IndAvgTrueRange):
    """Just a wrapped to create a pnael of atr band
    """

    def __init__(self,
                 data           : OHLCData,
                 period         : int = 5,
                 multiplier     : int = 3,
                 shift_ref_col  : ColName = Col.Close,
                 keep_tr_result : bool = True):

        self.shift_ref_col: ColName = shift_ref_col

        super().__init__(data,
                         period=period, multiplier=multiplier,
                         keep_tr_result=keep_tr_result)

    def _calc(self) -> pd.DataFrame:
        _df = super()._calc()

        _df['PlotUp'] = (
            _df[self.shift_ref_col.name]
            + self.multiplier * _df[Col.Ind.AvgTrueRange(self.period)]
        )
        _df['PlotDn'] = (
            _df[self.shift_ref_col.name]
            - self.multiplier * _df[Col.Ind.AvgTrueRange(self.period)]
        )

        return _df

    def make_addplot(self, plotter_args: dict, *args, **kwargs) -> list[dict]:


        return [
            mpf.make_addplot(
                self.df['PlotUp'],
                fill_between = {
                    'y1': self.df['PlotUp'].values,
                    'y2': self.df['PlotDn'].values,
                    'alpha': 0.3,
                    'color': 'dimgray',
                },
                alpha=0.,
                panel=plotter_args['main_panel'],
                label=f'{Col.Ind.AvgTrueRange(self.period)} Band'
            )
        ]


class IndSupertrend(_RollingMixin, _BandMixin, _BaseIndicator):
    """Add calculations related to Supertrend up/dn and the actual to
    display

    The best thing about supertrend it sends out accurate signals on
    precise time. The indicator is available on various trading platforms
    free of cost. The indicator offers quickest technical analysis to
    enable the intraday traders to make faster decisions. As said above, it
    is extremely simple to use and understand.

    However, the indicator is not appropriate for all the situations. It
    works when the market is trending. Hence it is best to use for
    short-term technical analysis. Supertrend uses only the two parameters
    of ATR and multiplier which are not sufficient under certain conditions
    to predict the accurate direction of the market.

    Understanding and identifying buying and selling signals in supertrend
    is the main crux for the intraday traders. Both the downtrends as well
    uptrends are represented by the tool. The flipping of the indicator
    over the closing price indicates signal. A buy signal is indicated in
    green color whereas sell signal is given as the indicator turns red. A
    sell signal occurs when it closes above the price. """

    def __init__(
            self,
            data          : OHLCData,
            period        : int           = 7,
            multiplier    : int           = 3,
            multiplier_dn : Optional[int] = None  # if None, the same as multiplier
    ):
        super().__init__(data,
                         period=period,
                         multiplier=multiplier, multiplier_dn=multiplier_dn)

    @property
    def _col_supertrend_name(self) -> str:
        if self._multi_dn == self._multi_up:
            return Col.Ind.SuperTrend.Final(self.period, self._multi_up)
        else:
            return Col.Ind.SuperTrend.Final(self.period, self._multi_up, self._multi_dn)

    def _calc(self) -> pd.DataFrame:
        period = self.period
        _multi_up = self._multi_up
        _multi_dn = self._multi_dn

        ind_atr = IndAvgTrueRange(self._data, period)
        _df = ind_atr.get_result()

        # NOTE - the rolling min is one way to use but that will introduce another paratermeter for the
        #        window size
        _df[Col.Ind.SuperTrend.Up(period, _multi_up)] = (
            0.5 * (_df[Col.High.cur] + _df[Col.Low.cur])
            + _multi_up * _df[Col.Ind.AvgTrueRange(period)]
        )#.rolling(period).min()
        _df[Col.Ind.SuperTrend.Dn(period, _multi_dn)] = (
            0.5 * (_df[Col.High.cur] + _df[Col.Low.cur])
            - _multi_dn * _df[Col.Ind.AvgTrueRange(period)]
        )#.rolling(period).max()

        supertrend, modes = self._adjust_base_boundary_for_supertrend(_df)

        # # TrendUp is the resistence
        _df[self._col_supertrend_name] = supertrend
        _df[Col.Ind.SuperTrend.Mode.name] = modes

        _df = _df[[
            self.tick_col,
            Col.Ind.TrueRange.name,
            Col.Ind.AvgTrueRange(period),
            Col.Ind.SuperTrend.Up(period, _multi_up),
            Col.Ind.SuperTrend.Dn(period, _multi_dn),
            self._col_supertrend_name,
            Col.Ind.SuperTrend.Mode.name
        ]]

        return _df

    def _adjust_base_boundary_for_supertrend(self, _df):
        """The upper line is the median price plus a multiple of Average
        True Range (ATR). Similarly, the lower line is the median price
        minus a multiple of the ATR. Similar to any trailing stop, the
        upper line follows the market lower and will never move higher once
        established. Conversely, the lower line follows the market higher
        and will never move lower once established.
        """

        period = self.period
        _multi_up = self._multi_up
        _multi_dn = self._multi_dn

        closes = _df[Col.Close.cur].to_list()
        base_ups = _df[Col.Ind.SuperTrend.Up(period, _multi_up)].to_list()
        base_dns = _df[Col.Ind.SuperTrend.Dn(period, _multi_dn)].to_list()

        modes = [np.nan] * len(_df)
        supertrend = np.full((len(_df),), np.nan)
        final_ups = np.full((len(_df),), np.nan)
        final_dns = np.full((len(_df),), np.nan)

        final_ups[period] = base_ups[period]
        final_dns[period] = base_dns[period]
        supertrend[period] = base_ups[period]
        # 1 for showing support, 0 for showing resistence
        modes[period] = 1  # just a random pick

        for idx in range(period+1, len(_df)):
            # The previous resistence is too conservative
            if closes[idx-1] > final_ups[idx-1]:
                final_ups[idx] = base_ups[idx]
            else:
                final_ups[idx] = min(base_ups[idx], final_ups[idx-1])
                
            # The previous resistence is too conservative
            if closes[idx-1] < final_dns[idx-1]:
                final_dns[idx] = base_dns[idx]
            else:
                final_dns[idx] = max(base_dns[idx], final_dns[idx-1])

            # Break support, switch to resistence mode
            if closes[idx] < final_dns[idx]:
                modes[idx] = 0
            # Break resistence, switch to support mode
            elif closes[idx] > final_ups[idx]:
                modes[idx] = 1
            # Otherwise, just keep current trend
            else:
                modes[idx] = modes[idx-1]

            supertrend[idx] = final_dns[idx] if modes[idx] == 1 else final_ups[idx]

        return supertrend, modes

    def make_addplot(
            self,
            plotter_args: dict,
            with_raw_atr_band: bool = False
    ) -> list[dict]:

        period = self.period
        _multi_up = self._multi_up
        _multi_dn = self._multi_dn


        kwargs = dict(
                type='line',
                width=1,
                panel=plotter_args['main_panel']
        )
        kwargs_up = {'color': 'r', **kwargs}
        kwargs_dn = {'color': 'lime', **kwargs}


        _df_up = self.df.copy()
        _df_up.loc[_df_up[Col.Ind.SuperTrend.Mode.name] == 1, self._col_supertrend_name] = pd.NA
        ups = _df_up[self._col_supertrend_name].values
        
        _df_dn = self.df.copy()
        _df_dn.loc[_df_dn[Col.Ind.SuperTrend.Mode.name] == 0, self._col_supertrend_name] = pd.NA
        dns = _df_dn[self._col_supertrend_name].values

        s2r_markers = np.full((len(dns)), np.nan)
        r2s_markers = np.full((len(dns)), np.nan)

        modes = self.df[Col.Ind.SuperTrend.Mode.name].values
        # Locate the transition point, add markes and fill in the gap
        for idx, mode in enumerate(modes[:-1]):
            if mode not in {0, 1}: continue  # first n points are NA

            if mode != modes[idx+1]:
                # support to resistence: fall breach the support, red line, red downward arrow above
                if mode == 1: 
                    ups[idx] = dns[idx]
                    # TODO - add markers
                    s2r_markers[idx+1] = ups[idx+1] * 1.05
                else:
                    # resistence to support: risk breach the resistence, green line, green upward arrow below
                    dns[idx] = ups[idx]
                    r2s_markers[idx+1] = dns[idx+1] * 0.95

        if with_raw_atr_band:
            kwargs['fill_between'] = {
                'y1': self.df[Col.Ind.SuperTrend.Up(period, _multi_up)].values,
                'y2': self.df[Col.Ind.SuperTrend.Dn(period, _multi_dn)].values,
                'alpha': 0.3,
                'color': 'dimgray'
            }

        return [
            mpf.make_addplot(ups, **kwargs_up),
            mpf.make_addplot(dns, **kwargs_dn),
            mpf.make_addplot(s2r_markers, type='scatter', markersize=200, color='r', marker='v'),
            mpf.make_addplot(r2s_markers, type='scatter', markersize=200, color='lime', marker='^'),
        ]


class IndStarcBand(_BandMixin, _BaseIndicator):
    """Commonly called STARC Bands, Stoller Average Range Channel Bands
    developed by Manning Stoller, are two bands that are applied above and
    below a simple moving average (SMA) of an asset's price. The upper band
    is created by adding the value of the average true range (ATR), or a
    multiple of it. The lower band is created by subtracting the value of
    the ATR from the SMA.
    
    The channel created by the bands can provide traders with ideas on when
    to buy or sell. During an overall uptrend, buying near the lower band
    and selling near the top band is favorable, for example. STARC bands
    can provide insight for both ranging and trending markets
    """
    

    def __init__(
            self,
            data          : OHLCData,
            period_sma    : int           = 5,
            period_atr    : int           = 15,
            multiplier    : int           = 3,
            multiplier_dn : Optional[int] = None  # if None, the same as multiplier
    ):
        """
        period: for SMA, usually between 5 and 10
        multiplier: for ATR scale, commonly take to be 2
        """
        self.period_sma = period_sma
        self.period_atr = period_atr
        super().__init__(data,
                         multiplier=multiplier, multiplier_dn=multiplier_dn)

    def _calc(self) -> pd.DataFrame:

        self._sma = IndSMA(self._data, self.period_sma)
        _sma_val = self._sma.df[Col.Ind.SMA(self.period_sma)].values

        self._atr = IndAvgTrueRange(self._data, self.period_atr, keep_tr_result=False)
        _atr_val = self._atr.df[Col.Ind.AvgTrueRange(self.period_atr)].values


        _df = self._df[[self.tick_col]].copy()
        _df[Col.Ind.STARC.SMA(self.period_sma)] = _sma_val
        _df[Col.Ind.STARC.ATR(self.period_atr)] = _atr_val
        _df[Col.Ind.STARC.Up(self.period_sma, self.period_atr)] = _sma_val + self._multi_up * _atr_val
        _df[Col.Ind.STARC.Dn(self.period_sma, self.period_atr)] = _sma_val - self._multi_dn * _atr_val

        return _df

    def make_addplot(self, plotter_args: dict,
                     *args,
                     with_sma: bool = False,
                     fill_band: bool = True,
                     **kwargs) -> list[dict]:
        kwargs = {
            'alpha': 0.,
            'label': Col.Ind.STARC.STARC(self.period_sma, self.period_atr),
            'secondary_y': False
        }

        if fill_band:
            kwargs['fill_between'] = {
                'y1': self.df[Col.Ind.STARC.Up(self.period_sma, self.period_atr)].values,
                'y2': self.df[Col.Ind.STARC.Dn(self.period_sma, self.period_atr)].values,
                'alpha': 0.3,
                'color': 'dimgray',
            }
        
        ret = [mpf.make_addplot(self.df[Col.Close.name], **kwargs)]

        if with_sma:
            # Ensure we have the same size
            self._sma._df = self._sma._df.merge(self.df[[self.tick_col]], how='inner', on=self.tick_col)
            ret += self._sma.make_addplot(plotter_args)
        
        return ret

class IndAroon(_RollingMixin, _BaseIndicator):
    """The Aroon indicator is a technical analysis tool used to identify
    trends and potential reversal points in the price movements of a
    security. It was developed by Tushar Chande in 1995. The term "Aroon" is
    derived from the Sanskrit word meaning "dawn's early light," symbolizing
    the beginning of a new trend.
    
    Aroon Up measures the number of periods since the highest price over a
    given period, expressed as a percentage of the total period.
    
    Aroon Up = (𝑛 − Periods since highest high) / 𝑛 × 100
    Aroon_Dn is defined similarly
    """

    def __init__(self,
                 data  : OHLCData,
                 period: int = 14):

        super().__init__(data, period=period)

    def _calc(self) -> pd.DataFrame:

        _df = self.df[[self.tick_col]].copy()

        highs = self.df[Col.High.name].values
        lows = self.df[Col.Low.name].values

        aroon_ups = np.full((len(_df),), np.nan)
        aroon_dns = np.full((len(_df),), np.nan)

        min_heap = []
        max_heap = []

        for i in range(self.period):
            heapq.heappush(min_heap, (lows[i], i))
            heapq.heappush(max_heap, (-highs[i], i))

        for i in range(self.period, len(_df)):

            # Remove index outside the window
            self.__clean_heap(min_heap, i - self.period)
            self.__clean_heap(max_heap, i - self.period)

            min_idx = min_heap[0][1]
            max_idx = max_heap[0][1]

            aroon_ups[i] = (self.period - (i - max_idx)) / (self.period) * 100
            aroon_dns[i] = (self.period - (i - min_idx)) / (self.period) * 100

            heapq.heappush(min_heap, (lows[i], i))
            heapq.heappush(max_heap, (-highs[i], i))
        
        _df[Col.Ind.Aroon.Up(self.period)] = aroon_ups
        _df[Col.Ind.Aroon.Dn(self.period)] = aroon_dns

        return _df

    @staticmethod
    def __clean_heap(heap, min_index):

        while heap and heap[0][1] < min_index:
            heapq.heappop(heap)
        
    @property
    def need_new_panel_num(self) -> bool:
        return True

    def make_addplot(self, plotter_args: dict, *args, **kwargs) -> list[dict]:

        # TODO - need to add color following style
        return [
            mpf.make_addplot(
                self.df[Col.Ind.Aroon.Up(self.period)],
                type='line', panel=plotter_args['new_panel_num'],
                color='lime', secondary_y = False
            ),
            mpf.make_addplot(
                self.df[Col.Ind.Aroon.Dn(self.period)],
                type='line', panel=plotter_args['new_panel_num'],
                color='red', secondary_y = False
            ),
            mpf.make_addplot(
                (self.df[Col.Ind.Aroon.Up(self.period)] - self.df[Col.Ind.Aroon.Dn(self.period)]),
                type='line', panel=plotter_args['new_panel_num'],
                color='k', linestyle='--', secondary_y = True
            )
        ]
        

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
    ):
        """
        period: for SMA, usually between 5 and 10
        multiplier: for ATR scale, commonly take to be 2
        """
        self.period_fast = period_fast
        self.period_slow = period_slow
        super().__init__(data)

    def _calc(self) -> pd.DataFrame:

        self._sma_fast = IndSMA(self._data, self.period_fast)
        _sma_fast_val = self._sma_fast.df[Col.Ind.SMA(self.period_fast)].values

        self._sma_slow = IndSMA(self._data, self.period_slow)
        _sma_slow_val = self._sma_slow.df[Col.Ind.SMA(self.period_slow)].values

        _df = self._df[[self.tick_col]].copy()
        _df[Col.Ind.AwesomeOscillator.Fast(self.period_fast)] = _sma_fast_val
        _df[Col.Ind.AwesomeOscillator.Slow(self.period_slow)] = _sma_slow_val
        _df[Col.Ind.AwesomeOscillator.AO(self.period_fast, self.period_slow)] = \
            _sma_fast_val - _sma_slow_val

        return _df

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

