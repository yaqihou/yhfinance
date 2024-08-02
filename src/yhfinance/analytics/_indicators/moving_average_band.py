# 
from typing import Optional

import pandas as pd

import mplfinance as mpf

from ..ohlc_cols import Col, ColName
from ..ohlc_processor import OHLCInterProcessor
from ..ohlc_data import OHLCData

from ._indicators_mixin import *
from ._base import _BaseIndicator
from .basics import IndSMA
from .volatility import IndTrueRange


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

        ind_tr = IndTrueRange(self._data, price_col=self.price_col)
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
    def values(self):
        return self.df[Col.Ind.AvgTrueRange(self.period)].values

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

    @property
    def values(self):
        return {
            'Up': self.df['PlotUp'].values,
            'Dn': self.df['PlotDn'].values,
        }


class IndBollingerBand(_RollingMixin, _BandMixin, _BaseIndicator):
    """The Awesome Oscillator is an indicator used to measure market
    momentum. AO calculates the difference of a 34 Period and 5 Period
    Simple Moving Averages. The Simple Moving Averages that are used are
    not calculated using closing price but rather each bar's midpoints. AO
    is generally used to affirm trends or to anticipate possible reversals.

    """
    
    def __init__(
            self,
            data          : OHLCData,
            period        : int           = 20,
            multiplier    : int           = 2,
            multiplier_dn : Optional[int] = None,
            price_col     : ColName       = Col.Close
    ):
        """
        period: for SMA, usually between 5 and 10
        multiplier: for std scale, commonly take to be 2
        """
        super().__init__(data,
                         period=period,
                         multiplier=multiplier,
                         multiplier_dn=multiplier_dn,
                         price_col=price_col)

    def _calc(self) -> pd.DataFrame:

        self._sma = IndSMA(self._data, self.period, price_col=self.price_col)
        _sma_val = self._sma.df[Col.Ind.SMA(self.period)].values

        self._std = self.df[self.price_col].rolling(self.period).std()
        _std_val = self._std.values

        _df = self._df[[self.tick_col]].copy()
        _df[Col.Ind.BollingerBand.SMA(self.period)] = _sma_val
        _df[Col.Ind.BollingerBand.Std(self.period)] = _std_val

        _col_up = Col.Ind.BollingerBand.Up(self.period, self._multi_up)
        _col_dn = Col.Ind.BollingerBand.Dn(self.period, self._multi_dn)
        _df[_col_up] = _sma_val + self._multi_up * _std_val
        _df[_col_dn] = _sma_val - self._multi_dn * _std_val

        return _df

    @property
    def values(self):
        return {
            'Up':  self.df[Col.Ind.BollingerBand.Up(self.period, self._multi_up)].values,
            'Dn':  self.df[Col.Ind.BollingerBand.Dn(self.period, self._multi_dn)].values
        }

    @property
    def need_new_panel_num(self) -> bool:
        return False

    def make_addplot(self, plotter_args: dict,
                     *args,
                     **kwargs) -> list[dict]:

        args = (self._multi_up,) if self._multi_up == self._multi_dn else (self._multi_up, self._multi_dn)
        kwargs = {
            'label': Col.Ind.BollingerBand.BB(self.period, *args),
            'secondary_y': False,
            'fill_between': {
                'y1': self.df[Col.Ind.BollingerBand.Up(self.period, self._multi_up)].values,
                'y2': self.df[Col.Ind.BollingerBand.Dn(self.period, self._multi_dn)].values,
                'alpha': 0.3,
                'color': 'dimgray',
            }
        }

        return [
            mpf.make_addplot(self.df[Col.Ind.BollingerBand.SMA(self.period)], **kwargs)
        ]


# TODO - need to support log return
class IndBollingerBandModified(IndBollingerBand):
    """The only difference between ordinary BollingerBand is that the std is calculated
    based on the past n periods relative return 
    """
    
    def __init__(
            self,
            data          : OHLCData,
            period        : int           = 20,
            multiplier    : int           = 2,
            multiplier_dn : Optional[int] = None,
            price_col     : ColName       = Col.Close
    ):
        """
        period: for SMA, usually between 5 and 10
        multiplier: for std scale, commonly take to be 2
        """
        super().__init__(data,
                         period=period,
                         multiplier=multiplier,
                         multiplier_dn=multiplier_dn,
                         price_col=price_col)


    def _calc(self) -> pd.DataFrame:

        self._sma = IndSMA(self._data, self.period, price_col=self.price_col)
        _sma_val = self._sma.df[Col.Ind.SMA(self.period)].values

        _inter_processor = OHLCInterProcessor(self._data)
        _inter_processor.add_all_return()
        _df_inter = _inter_processor.get_result()
        _rtn_col = Col.Inter.get_return_col(self.price_col)
        _df_inter['RtnStd'] = _df_inter[_rtn_col.rtn].rolling(self.period).std()

        _df = self._df[[self.tick_col]].copy()
        _df = _df.merge(_df_inter[[self.tick_col, 'RtnStd']], how='left', on=self.tick_col)

        _df[Col.Ind.BollingerBandModified.SMA(self.period)] = _sma_val
        _df[Col.Ind.BollingerBandModified.Std(self.period)] = (
            self._df[self.price_col] * 0.01 * _df['RtnStd'])
        _std_val = _df[Col.Ind.BollingerBandModified.Std(self.period)].values

        _col_up = Col.Ind.BollingerBandModified.Up(self.period, self._multi_up)
        _col_dn = Col.Ind.BollingerBandModified.Dn(self.period, self._multi_dn)
        _df[_col_up] = _sma_val + self._multi_up * _std_val
        _df[_col_dn] = _sma_val - self._multi_dn * _std_val

        return _df

    def make_addplot(self, plotter_args: dict,
                     *args,
                     **kwargs) -> list[dict]:

        args = (self._multi_up,) if self._multi_up == self._multi_dn else (self._multi_up, self._multi_dn)
        kwargs = {
            'label': Col.Ind.BollingerBandModified.BB(self.period, *args),
            'secondary_y': False,
            'fill_between': {
                'y1': self.df[Col.Ind.BollingerBandModified.Up(self.period, self._multi_up)].values,
                'y2': self.df[Col.Ind.BollingerBandModified.Dn(self.period, self._multi_dn)].values,
                'alpha': 0.3,
                'color': 'dimgray',
            }
        }

        return [
            mpf.make_addplot(self.df[Col.Ind.BollingerBandModified.SMA(self.period)], **kwargs)
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

    @property
    def values(self):
        return {
            'Up': self.df[Col.Ind.STARC.Up(self.period_sma, self.period_atr)].values,
            'Dn': self.df[Col.Ind.STARC.Dn(self.period_sma, self.period_atr)].values
        }

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


# class IndTriMovingAverage(_BaseIndicator):

#     def __init__(self,
#                  data: OHLCData,
#                  periods: list[int] | tuple[int, ...] = 
#                  price_col: ColName = Col.Close):
#         super().__init__(data, price_col)
