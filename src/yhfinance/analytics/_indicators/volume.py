import numpy as np
import pandas as pd

import mplfinance as mpf

from ..ohlc_cols import Col, ColName
from ..ohlc_data import OHLCData

from ._indicators_mixin import *
from ._base import _BaseIndicator


class IndMoneyFlowIndex(_RollingMixin, _BaseIndicator):
    """The Money Flow Index (MFI) is a technical oscillator that uses
    price and volume data for identifying overbought or oversold signals in
    an asset. It can also be used to spot divergences which warn of a trend
    change in price. The oscillator moves between 0 and 100.

    The Money Flow Index (MFI) is a technical indicator that generates
    overbought or oversold signals using both prices and volume data.

    An MFI reading above 80 is considered overbought and an MFI reading
    below 20 is considered oversold, although levels of 90 and 10 are also
    used as thresholds.

    A divergence between the indicator and price is noteworthy. For
    example, if the indicator is rising while the price is falling or flat,
    the price could start rising.

    """
    
    def __init__(
            self,
            data          : OHLCData,
            period        : int           = 14,
            price_col     : ColName       = Col.Typical
    ):
        super().__init__(data,
                         period=period,
                         price_col=price_col)


    def _calc(self) -> pd.DataFrame:

        _df = self._df[[self.tick_col]].copy()

        flows =  (self._df[self.price_col] * self._df[Col.Vol]).values
        _df[Col.Ind.MFI.Flow.name] = flows

        prices = self._df[self.price_col].values

        pos = 0.
        neg = 0.
        pos_vals = np.full((len(flows), ), np.nan)
        neg_vals = np.full((len(flows), ), np.nan)
        for idx, flow in enumerate(flows[1:self.period+1], 1):
            if prices[idx] > prices[idx] - 1:
                pos += flow
            else:
                neg += flow
            pos_vals[idx] = pos
            neg_vals[idx] = neg

        for idx, flow in enumerate(flows[self.period+1:], self.period+1):

            prev_idx = idx - self.period
            # Remove the contribution from out-of-window sample
            if prices[prev_idx] > prices[prev_idx-1]:
                pos -= flows[prev_idx]
            else:
                neg -= flows[prev_idx]

            # Add the contribution from newly added sample
            if prices[idx] > prices[idx-1]:
                pos += flow
            else:
                neg += flow

            pos_vals[idx] = pos
            neg_vals[idx] = neg

        with np.errstate(divide='ignore'):
            ratios = np.divide(pos_vals, neg_vals)
        _df[Col.Ind.MFI.Pos(self.period)] = pos_vals
        _df[Col.Ind.MFI.Neg(self.period)] = neg_vals
        _df[Col.Ind.MFI.Ratio(self.period)] = ratios
        _df[Col.Ind.MFI.MFI(self.period)] = 100 - (100 / (1 + ratios))

        return _df

    @property
    def values(self):
        return self._df[Col.Ind.MFI.MFI].values
    
    @property
    def need_new_panel_num(self) -> bool:
        return True

    def make_addplot(self, plotter_args: dict,
                     *args,
                     **kwargs) -> list[dict]:

        return [
            mpf.make_addplot(
                self.df[Col.Ind.MFI.MFI(self.period)],
                type='line', panel=plotter_args['new_panel_num'],
                label=Col.Ind.MFI.MFI(self.period)
            )
        ]
