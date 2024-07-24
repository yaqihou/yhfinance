# 
from collections import defaultdict
from typing import Optional

from matplotlib.pyplot import fill_between
import numpy as np
import pandas as pd

import mplfinance as mpf

import abc
from ._base import _OHLCBase
from .const import Col, ColIntra, ColName, _T_RSI
from .ohlc_processor import OHLCInterProcessor


class _BaseIndicator(_OHLCBase, abc.ABC):

    # TODO - need to implement for each indicator
    _category: str = "undefined"

    def __init__(self, df: pd.DataFrame, tick_col: str = Col.Date.name):
        super().__init__(df, tick_col)

        self._calc()

    @abc.abstractmethod
    def _calc(self) -> None:
        """Implement the calculation and populate the results to self._df"""

    @abc.abstractmethod
    def make_addplot(self, plotter_args: dict, *args, **kwargs) -> list[dict]:
        """Return a list of panel definition returned by mpf.make_addplot"""

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
                print(f'Warning: found existing RSI result col {col}, dropping it')
                drop_cols.append(col)
        
        if drop_cols:
            self._df = self._df.drop(columns=drop_cols)
        self._df = self._df.merge(df, how='left', on=self.tick_col)


class IndMACD(_BaseIndicator):

    def __init__(
            self,
            df                : pd.DataFrame,
            tick_col          : str     = Col.Date.name,
            short_term_window : int     = 12,
            long_term_window  : int     = 26,
            signal_window     : int     = 9,
            price_col         : ColName = Col.Close
    ):

        self.short_term_window = short_term_window
        self.long_term_window  = long_term_window
        self.signal_window     = signal_window
        self.price_col         = price_col

        super().__init__(df, tick_col)

    def _calc(self):
        
        ewm_short = self._df[self.price_col.name].ewm(
            span=self.short_term_window, adjust=False).mean()
        ewm_long = self._df[self.price_col.name].ewm(
            span=self.long_term_window, adjust=False).mean()

        macd = ewm_short - ewm_long
        signal = macd.ewm(span=self.signal_window, adjust=False).mean()

        df = self._df[[self.tick_col]].copy()
        
        df[Col.Ind.Momentum.MACD.EMA12.name] = ewm_short
        df[Col.Ind.Momentum.MACD.EMA26.name] = ewm_long
        df[Col.Ind.Momentum.MACD.MACD(
            self.short_term_window, self.long_term_window, self.signal_window
        )] = macd

        df[Col.Ind.Momentum.MACD.Signal(
            self.short_term_window, self.long_term_window, self.signal_window)] = signal

        self._add_result_safely(df)

    # TODO - could add a argument to control this in init
    @property
    def need_new_panel_num(self) -> bool:
        return True

    def make_addplot(self, plotter_args: dict, *args, **kwargs) -> list[dict]:

        main_panel = plotter_args['main_panel']
        new_panel_num = plotter_args['new_panel_num']

        _col_macd = Col.Ind.Momentum.MACD.MACD(
            self.short_term_window, self.long_term_window, self.signal_window)
        _col_signal = Col.Ind.Momentum.MACD.Signal(
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
                self.df[Col.Ind.Momentum.MACD.EMA12.name], linestyle='--',
                panel=main_panel, label='MACD-EMA12'),
            mpf.make_addplot(
                self.df[Col.Ind.Momentum.MACD.EMA26.name], linestyle=':',
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


class _IndRSI(_BaseIndicator):

    def __init__(
            self,
            df        : pd.DataFrame,
            tick_col  : str                 = Col.Date.name,
            n         : int                 = 14,
            threshold : tuple[float, float] = (30, 70)
    ):
        self.n = n
        self.threshold = threshold
        super().__init__(df, tick_col)

    @property
    def need_new_panel_num(self) -> bool:
        return True

    def _get_gl_for_rsi(self):

        _df = OHLCInterProcessor(self.df, tick_offset=-1).add_close_return().get_result()
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

    def _assign_rsi_result(self, df):
        
        df[self.rsi_col.AvgGain(self.n)] = self.ewm_ups
        df[self.rsi_col.AvgLoss(self.n)] = self.ewm_dns
        df[self.rsi_col.RS(self.n)] = self.ewm_ups / self.ewm_dns
        df[self.rsi_col.RSI(self.n)] = 100 - (100 / (1 + self.ewm_ups / self.ewm_dns))

        self._add_result_safely(df)

        return

    def make_addplot(self, plotter_args: dict, *args, **kwargs) -> list[dict]:

        new_panel_num = plotter_args['new_panel_num']

        _df_threshold = pd.DataFrame.from_dict({
            'upper': [max(self.threshold)] * len(self.df),
            'lower': [min(self.threshold)] * len(self.df)
        })
        
        return [
            mpf.make_addplot(self.df[self.rsi_col.RSI(self.n)],
                             type='line', color='r',
                             label=self.rsi_col.RSI(self.n),
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
        return Col.Ind.Momentum.RSIWilder

    def _calc(self) -> None:
        _df, ups, dns = self._get_gl_for_rsi()
        n = self.n

        ups.iloc[n] = ups.iloc[:n].mean()
        ups.iloc[:n] = pd.NA
        dns.iloc[n] = dns.iloc[:n].mean()
        dns.iloc[:n] = pd.NA

        alpha = 1 / n
        self.ewm_ups = ups.ewm(alpha=alpha, ignore_na=True, adjust=False).mean()
        self.ewm_dns = dns.ewm(alpha=alpha, ignore_na=True, adjust=False).mean()

        self._assign_rsi_result(_df)


class IndEmaRSI(_IndRSI):
    """The main difference from wilder's original is that instead of SMMA,
    we used a EMA so that the first n obs are NOT nan"""

    @property
    def rsi_col(self) -> _T_RSI:
        return Col.Ind.Momentum.RSIEma

    def _calc(self) -> None:
        _df, ups, dns = self._get_gl_for_rsi()

        alpha = 1 / self.n
        self.ewm_ups = ups.ewm(alpha=alpha, adjust=False).mean()
        self.ewm_dns = dns.ewm(alpha=alpha, adjust=False).mean()

        self._assign_rsi_result(_df)


class IndCutlerRSI(_IndRSI):
    """Cutler's RSI variation is based on SMA, to overcome the so-called 'Data Length Dependency'"""

    @property
    def rsi_col(self) -> _T_RSI:
        return Col.Ind.Momentum.RSICutler

    def _calc(self) -> None:
        _df, ups, dns = self._get_gl_for_rsi()

        self.ewm_ups = ups.rolling(self.n).mean()
        self.ewm_dns = dns.rolling(self.n).mean()

        self._assign_rsi_result(_df)


class IndTrueRange(_BaseIndicator):

    def __init__(
            self,
            df       : pd.DataFrame,
            tick_col : str = Col.Date.name,
            period   : int = 14
    ):
    
        self.period = period
        super().__init__(df, tick_col)

    def _calc(self) -> None:
        _df = OHLCInterProcessor(self.df, tick_offset=-1)._df_offset

        # TR = max[(H-L), abs(H-Cp), abs(L-Cp)]
        _df[Col.Ind.Band.TrueRange.name] = pd.concat([
            _df[Col.High.cur] - _df[Col.Low.cur],
            (_df[Col.High.cur] - _df[Col.Close.sft]).abs(),
            (_df[Col.Low.cur] - _df[Col.Close.sft]).abs()
            ], axis=1).max(axis=1)

        self._add_result_safely(_df)

    def make_addplot(self, plotter_args: dict, *args, **kwargs) -> list[dict]:
        raise NotImplementedError()
    

class IndAvgTrueRange(_BaseIndicator):

    def __init__(
            self,
            df       : pd.DataFrame,
            tick_col : str = Col.Date.name,
            period   : int = 14,
            keep_tr_result: bool = True
    ):
    
        self.period = period
        self.keep_tr_result = keep_tr_result

        super().__init__(df, tick_col)

    def _calc(self) -> None:
        # ATR_curr = ATR_prev * (n-1) / n + TR_curr * (1/n)
        # i.e. a EWM with alpha = 1/n
        # initial condition is ATR_0 = mean(sum(TR_n))
        period = self.period

        ind_tr = IndTrueRange(self.df, self.tick_col, period)
        _df = ind_tr.get_result()

        _tr = _df[Col.Ind.Band.TrueRange.name].copy()
        _tr.iloc[period] = _tr.iloc[:period].mean()
        _tr.iloc[:period] = pd.NA

        alpha = 1 / period
        _df[Col.Ind.Band.AvgTrueRange(period)] = _tr.ewm(alpha=alpha, ignore_na=True, adjust=False).mean()

        if not self.keep_tr_result:
            _df = _df[[self.tick_col, Col.Ind.Band.AvgTrueRange(period)]]

        self._add_result_safely(_df)


    def make_addplot(self, plotter_args: dict, *args, **kwargs) -> list[dict]:
        raise NotImplementedError()


class IndSupertrend(_BaseIndicator):
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
            df            : pd.DataFrame,
            tick_col      : str           = Col.Date.name,
            period        : int           = 7,
            multiplier    : int           = 3,
            multiplier_dn : Optional[int] = None  # if None, the same as multiplier
    ):
        self.period         = period
        self._multi_up      = multiplier
        self._multi_dn      = multiplier_dn or multiplier

        super().__init__(df, tick_col)

    @property
    def _col_supertrend_name(self) -> str:
        if self._multi_dn == self._multi_up:
            return Col.Ind.Band.SuperTrend.Final(self.period, self._multi_up)
        else:
            return Col.Ind.Band.SuperTrend.Final(self.period, self._multi_up, self._multi_dn)

    def _calc(self) -> None:
        period = self.period
        _multi_up = self._multi_up
        _multi_dn = self._multi_dn

        ind_atr = IndAvgTrueRange(self.df, self.tick_col, period)
        _df = ind_atr.get_result()

        # NOTE - the rolling min is one way to use but that will introduce another paratermeter for the
        #        window size
        _df[Col.Ind.Band.SuperTrend.Up(period, _multi_up)] = (
            0.5 * (_df[Col.High.cur] + _df[Col.Low.cur])
            + _multi_up * _df[Col.Ind.Band.AvgTrueRange(period)]
        )#.rolling(period).min()
        _df[Col.Ind.Band.SuperTrend.Dn(period, _multi_dn)] = (
            0.5 * (_df[Col.High.cur] + _df[Col.Low.cur])
            - _multi_dn * _df[Col.Ind.Band.AvgTrueRange(period)]
        )#.rolling(period).max()

        supertrend, modes = self._adjust_base_boundary_for_supertrend(_df)

        # # TrendUp is the resistence
        _df[self._col_supertrend_name] = supertrend
        _df[Col.Ind.Band.SuperTrend.Mode.name] = modes

        _df = _df[[
            self.tick_col,
            Col.Ind.Band.TrueRange.name,
            Col.Ind.Band.AvgTrueRange(period),
            Col.Ind.Band.SuperTrend.Up(period, _multi_up),
            Col.Ind.Band.SuperTrend.Dn(period, _multi_dn),
            self._col_supertrend_name,
            Col.Ind.Band.SuperTrend.Mode.name
        ]]

        self._add_result_safely(_df)

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
        base_ups = _df[Col.Ind.Band.SuperTrend.Up(period, _multi_up)].to_list()
        base_dns = _df[Col.Ind.Band.SuperTrend.Dn(period, _multi_dn)].to_list()

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
        _df_up.loc[_df_up[Col.Ind.Band.SuperTrend.Mode.name] == 1, self._col_supertrend_name] = pd.NA
        ups = _df_up[self._col_supertrend_name].values
        
        _df_dn = self.df.copy()
        _df_dn.loc[_df_dn[Col.Ind.Band.SuperTrend.Mode.name] == 0, self._col_supertrend_name] = pd.NA
        dns = _df_dn[self._col_supertrend_name].values

        s2r_markers = np.full((len(dns)), np.nan)
        r2s_markers = np.full((len(dns)), np.nan)

        modes = self.df[Col.Ind.Band.SuperTrend.Mode.name].values
        print(modes)
        # Locate the transition point, add markes and fill in the gap
        for idx, mode in enumerate(modes[:-1]):
            if mode not in {0, 1}: continue  # first n points are NA

            if mode != modes[idx+1]:
                print(idx, mode, modes[idx+1])
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
            fill_between = {
                'y1': self.df[Col.Ind.Band.SuperTrend.Up(period, _multi_up)].values,
                'y2': self.df[Col.Ind.Band.SuperTrend.Dn(period, _multi_dn)].values,
                'alpha': 0.3,
                'color': 'dimgray'
            }

        return [
            mpf.make_addplot(ups, fill_between=fill_between, **kwargs_up),
            mpf.make_addplot(dns, **kwargs_dn),
            mpf.make_addplot(s2r_markers, type='scatter', markersize=200, color='r', marker='v'),
            mpf.make_addplot(r2s_markers, type='scatter', markersize=200, color='lime', marker='^'),
        ]

    
