# 
import heapq
from typing import Optional

import numpy as np
import pandas as pd

import mplfinance as mpf

import abc
from ..ohlc_cols import Col
from ..ohlc_data import OHLCData

from ._indicators_mixin import *
from ._base import _BaseIndicator
from .moving_average_band import IndAvgTrueRange


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

    @property
    def values(self):
        return {
            'Up': self.df[Col.Ind.Aroon.Up(self.period)].values,
            'Dn': self.df[Col.Ind.Aroon.Dn(self.period)].values
        }

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

    @property
    def values(self):
        return self.df[self._col_supertrend_name].values

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
