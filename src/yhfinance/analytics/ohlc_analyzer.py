

from collections import namedtuple
from typing import Literal

import pandas as pd
import datetime as dt

import seaborn as sns
import matplotlib.pyplot as plt

from yhfinance.analytics.const import ColName

from ._base import _OHLCBase
from .const import Col, ColName
from .ohlc_processor import OHLCFixedWindowProcessor


class OHLCSimpleStrategy(_OHLCBase):
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
        _df['Rtn'] = _df['Gl'] / _df['Buy'] * 100

        return _df

    def buy_at_first_day_gl(self, buy_at: ColName = Col.Open):
        """Return the gain/loss if buy at the first day"""

        _df = self._get_gl_rtn(
            buy_at=self._df.iloc[0, :].loc[buy_at.name],
            sell_at=Col.High.name)

        return _df

    def max_gain(self, buy_at: ColName = Col.Low,  # Could also be Close / Open
                 sell_at: ColName = Col.High,
                 allow_intraday_trading: bool = False,  # TODO - if buy and high could happen at the same day
                 return_raw: bool = False
                 ):
        """Return the max possible gain during this period, note that max
        gain could happen differently from the max return"""

        _df = self._get_gl_rtn(buy_at=buy_at.name + 'RollingMin', sell_at=sell_at.name)
        if return_raw:
            return _df
        raise NotImplementedError("TODO")

    def max_loss(self, buy_at: ColName = Col.High,
                 sell_at: ColName = Col.Low,
                 allow_intraday_trading: bool = False,  # TODO - if buy and high could happen at the same day
                 return_raw: bool = False
                 ):
        """Return the max possible loss during this period"""
        _df = self._get_gl_rtn(buy_at=buy_at.name + 'RollingMax', sell_at=sell_at.name)
        if return_raw:
            return _df
        raise NotImplementedError("TODO")


class OHLCSeasonalityAnalyzer(_OHLCBase):

    def __init__(
            self,
            df: pd.DataFrame,
            tick_col: str = Col.Date.name,
            group_cols: list[OHLCFixedWindowProcessor._T_CAL_INFO_COLS] = ['Year', 'Month']
    ):
        super().__init__(df, tick_col)
        self.processor = OHLCFixedWindowProcessor(df, tick_col=tick_col)

        self.group_cols = group_cols

        # Prepare the summary table
        # _summary_dict = {k: [] for k in group_cols}
        self._uniq_summary_key = sorted(self.df_processed[group_cols].apply(tuple, axis=1).unique())
        # for x in self._uniq_summary_key:
        #     for k, v in zip(group_cols, x):
        #         _summary_dict[k].append(v)
        self._df_summary = pd.DataFrame.from_dict({'_key': self._uniq_summary_key})

        # self.df_slice_dict = {
        #     key: self.processor.get_slice(**{x.lower(): y for x, y in zip(group_cols, key)})
        #     for key in self._uniq_summary_key
        # }


    def plot_details(self, figsize=(30, 24)):

        fig, axes = plt.subplots(len(self.processor.years), figsize=figsize)

        # for idx_row, year in enumerate(year_list):
        #     for idx_month, month in enumerate(month_list):

        #         ax = axes[idx_month][idx_year]
        #         sns.lineplot(self.df_slice_dict[], x='Date', y='Close', ax=ax)
        #         ax.set_xticks([])
        #         ax.set_xlabel("")
        #         ax.set_ylabel("")

        #         # _ax = ax.twinx()
        #         # sns.lineplot(_df, x='Date', y='52 Wk High', ax=ax, color='r')
        #         # sns.lineplot(_df, x='Date', y='52 Wk Low', ax=ax, color='r', ls='--')
        #         # _ax.set_xticks([])
        #         # _ax.set_xlabel("")
        #         # _ax.set_ylabel("")

        #         ax.set_title(f"{year}-{month:02d}")
        plt.tight_layout()
        plt.show()

    @property
    def df_processed(self):
        return self.processor.df

    @property
    def df_summary(self):
        _df = self._df_summary.copy()
        _df[self.group_cols] = pd.DataFrame(_df['_key'].tolist(), index=_df.index)
        _df = _df.drop(columns={'_key'})
        # Reorder the columns
        cols = self.group_cols + list(filter(lambda x: x not in self.group_cols, _df.columns))
        return _df[cols]
    
    def apply_simple_strategies(self):

        for _key, _df_slice in self.df_processed.groupby(self.group_cols):
            _df_slice = _df_slice.reset_index(drop=True)
            mask = self._df_summary['_key'] == _key

            start = OHLCSimpleStrategy(_df_slice)

            _df = start.buy_at_first_day_gl()
            self._df_summary.loc[mask, 'BuyD01MaxReturn'] = _df['Rtn'].max()
            self._df_summary.loc[mask, 'BuyD01SellDay'] = _df['Rtn'].argmax()

            _buy_at = Col.Low
            _sell_at = Col.High

            _df = start.max_gain(return_raw=True)
            _sell_day = _df['Gl'].argmax()
            _buy_day = _df_slice[_df_slice[_buy_at.name] == _df.iloc[_sell_day, 0]].index[0]
            self._df_summary.loc[mask, 'MaxGain'] = _df['Gl'].max()
            self._df_summary.loc[mask, 'MaxGainBuyDay'] = _buy_day
            self._df_summary.loc[mask, 'MaxGainSellDay'] = _sell_day

            _sell_day = _df['Rtn'].argmax()
            _buy_day = _df_slice[_df_slice[_buy_at.name] == _df.iloc[_sell_day, 0]].index[0]
            self._df_summary.loc[mask, 'MaxGainRtn'] = _df['Rtn'].max()
            self._df_summary.loc[mask, 'MaxGainRtnBuyDay'] = _buy_day
            self._df_summary.loc[mask, 'MaxGainRtnSellDay'] = _sell_day

            # -------------------------

            _buy_at = Col.High
            _sell_at = Col.Low

            _df = start.max_loss(return_raw=True)
            _sell_day = _df['Gl'].argmin()
            _buy_day = _df_slice[_df_slice[_buy_at.name] == _df.iloc[_sell_day, 0]].index[0]
            self._df_summary.loc[mask, 'MaxLoss'] = _df['Gl'].min()
            self._df_summary.loc[mask, 'MaxLossBuyDay'] = _buy_day
            self._df_summary.loc[mask, 'MaxLossSellDay'] = _sell_day

            _df = start.max_loss(return_raw=True)
            _sell_day = _df['Rtn'].argmin()
            _buy_day = _df_slice[_df_slice[_buy_at.name] == _df.iloc[_sell_day, 0]].index[0]
            self._df_summary.loc[mask, 'MaxLossRtn'] = _df['Rtn'].min()
            self._df_summary.loc[mask, 'MaxLossRtnBuyDay'] = _buy_day
            self._df_summary.loc[mask, 'MaxLossRtnSellDay'] = _sell_day

        return


class OHLCStreakAnalyzer(_OHLCBase):

    # TODO - add sanity check to make sure the columns have streak to analysis
    pass


class OHLCVolatilityAnalyzer(_OHLCBase):

    # TODO - add sanity check to make sure the columns have streak to analysis
    pass
