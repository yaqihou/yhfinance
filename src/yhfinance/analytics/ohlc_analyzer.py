

from dataclasses import dataclass
from typing import Literal, get_args, Optional

from tqdm import tqdm
import numpy as np
import pandas as pd
import datetime as dt

import seaborn as sns
import matplotlib.pyplot as plt
import mplfinance as mpf

from yhfinance.analytics.const import ColName

from ._base import _OHLCBase
from .const import Col, ColName
from .ohlc_processor import OHLCFixedWindowProcessor
from .plotter import OHLCPlotter


@dataclass
class SimpleStratRes:

    gl: float
    rtn: float
    buy_day: int
    sell_day: int
    name: str

    @property
    def col_gl(self):
        return self.name + 'Gl'

    @property
    def col_rtn(self):
        return self.name + 'Rtn'

    @property
    def col_buy_day(self):
        return self.name + 'BuyDay'

    @property
    def col_sell_day(self):
        return self.name + 'SellDay'

    def to_dict(self):
        return {
            self.col_gl: self.gl,
            self.col_rtn: self.rtn,
            self.col_buy_day: self.buy_day,
            self.col_sell_day: self.sell_day
        }

_T_VALID_SIMPLE_STRAT = Literal['buy_n_hold', 'buy_d01_gain', 'buy_d01_loss',
                                'max_gain', 'max_loss',
                                'max_loss_in_gl', 'max_loss_in_rtn']
class OHLCSimpleStrategy(_OHLCBase):
    """Applying a series of strategy on the input df.
    """

    def __init__(self, df: pd.DataFrame, tick_col: str = Col.Date.name):
        super().__init__(df, tick_col)

        # Make sure the index is starting from 0
        self._df = self._df.reset_index(drop=True)

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
        self._df_rolling = self._df_rolling.merge(_df.reset_index(), on=self.tick_col).reset_index(drop=True)

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

        # Debug
        # assert _df['Gl'].argmax() == _df['Rtn'].argmax()
        # NOTE - If using rollingMax, like when calculating max loss, this is not guaranteed
        # assert _df['Gl'].argmin() == _df['Rtn'].argmin()

        return _df

    def buy_and_hold(
            self,
            buy_at_col: ColName = Col.Open,
            sell_at_col: ColName = Col.Close):
        """Retrun the Gl/Rtn if buy at first day and hold until the last day
        """

        _df = self._get_gl_rtn(
            buy_at=self._df.iloc[0, :].loc[buy_at_col.name],
            sell_at=self._df.iloc[-1, :].loc[sell_at_col.name],
        )

        assert len(_df['Gl'].unique()) == 1
        assert len(_df['Rtn'].unique()) == 1
        
        _gl = _df.iloc[0, 2]
        _rtn = _df.iloc[0, 3]
        _buy_day = 0
        _sell_day = len(_df) - 1

        return SimpleStratRes(
            gl=_gl,
            rtn = _rtn,
            buy_day = _buy_day,
            sell_day = _sell_day,
            name = f'BuyNHold+{buy_at_col.name}-{sell_at_col.name}'
        )
        
    def buy_at_first_day(
            self,
            buy_at_col: ColName = Col.Open,
            sell_at_col: ColName = Col.Close,
            result_type: Literal['gain', 'loss'] = 'gain'):
        """Return the gain/loss if buy at the first day"""

        _df = self._get_gl_rtn(
            buy_at=self._df.iloc[0, :].loc[buy_at_col.name],
            sell_at=sell_at_col.name)

        _buy_day = 0
        if result_type == 'gain':
            # Since 
            _gl = _df['Gl'].max()
            _rtn = _df['Rtn'].max()
            _sell_day = _df['Gl'].argmax()
        else:
            _gl = _df['Gl'].min()
            _rtn = _df['Rtn'].min()
            _sell_day = _df['Gl'].argmin()

        return SimpleStratRes(
            gl=_gl,
            rtn=_rtn,
            buy_day=_buy_day,
            sell_day=_sell_day,
            name = f'BuyD01{result_type.capitalize()}+{buy_at_col.name}-{sell_at_col.name}'
        )

    def _infer_buy_day(self,
                       df_gl_rtn: pd.DataFrame,
                       sell_day: int,
                       buy_at: str):

        # In df_gl_rtn, Buy column is the actual price, which could be a constant or a rolling 
        _buy_day = self._df_rolling[self._df_rolling[buy_at] == df_gl_rtn.iloc[sell_day, 0]].index[0]
        return _buy_day

    def max_gain(self,
                 buy_at_col: ColName = Col.Low,  # Could also be Close / Open
                 sell_at_col: ColName = Col.High,
                 allow_intraday_trading: bool = False,  # TODO - if buy and high could happen at the same day
                 return_raw: bool = False
                 ):
        """Return the max possible gain during this period, note that max
        gain could happen differently from the max return"""

        _buy_at = buy_at_col.name + 'RollingMin'
        _df = self._get_gl_rtn(buy_at=_buy_at, sell_at=sell_at_col.name)
        if return_raw:
            return _df

        _gl = _df['Gl'].max()
        _rtn = _df['Rtn'].max()
        _sell_day = _df['Gl'].argmax()
        _buy_day = self._infer_buy_day(_df, _sell_day, _buy_at)

        return SimpleStratRes(
            gl=_gl,
            rtn = _rtn,
            buy_day = _buy_day,
            sell_day = _sell_day,
            name = f'MaxGain+{buy_at_col.name}-{sell_at_col.name}'
        )

    def max_loss(self,
                 buy_at_col: ColName = Col.High,
                 sell_at_col: ColName = Col.Low,
                 allow_intraday_trading: bool = False,  # TODO - if buy and high could happen at the same day
                 # The min rtn could be different from min gl as the ref buy price is rollingMax
                 measure_type: Literal['gl', 'rtn'] = 'gl',
                 return_raw: bool = False):
        """Return the max possible loss during this period"""
        _buy_at = buy_at_col.name + 'RollingMax'
        _df = self._get_gl_rtn(buy_at=_buy_at, sell_at=sell_at_col.name)
        if return_raw:
            return _df

        if measure_type == 'gl':
            _gl = _df['Gl'].min()
            _sell_day = _df['Gl'].argmin()
            _rtn = _df.iloc[_sell_day, :].loc['Rtn']
        else:
            _rtn = _df['Rtn'].min()
            _sell_day = _df['Gl'].argmin()
            _gl = _df.iloc[_sell_day, :].loc['Gl']
        _buy_day = self._infer_buy_day(_df, _sell_day, _buy_at)

        return SimpleStratRes(
            gl=_gl,
            rtn = _rtn,
            buy_day = _buy_day,
            sell_day = _sell_day,
            name = f'MaxLoss{measure_type.capitalize()}+{buy_at_col.name}-{sell_at_col.name}'
        )


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
        self._uniq_group_key = sorted(self.df_processed[group_cols].apply(tuple, axis=1).unique())
        # for x in self._uniq_summary_key:
        #     for k, v in zip(group_cols, x):
        #         _summary_dict[k].append(v)
        self._df_grouped = pd.DataFrame.from_dict({'_key': self._uniq_group_key})

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
    def df_grouped(self):
        _df = self._df_grouped.copy()
        _df[self.group_cols] = pd.DataFrame(_df['_key'].tolist(), index=_df.index)
        _df = _df.drop(columns=['_key'])
        # Reorder the columns
        cols = self.group_cols + list(filter(lambda x: x not in self.group_cols, _df.columns))
        return _df[cols]
    
    def apply_simple_strategies(
            self,
            # TODO - think about how to easily control the strategy in use
            selected_strategies: list[_T_VALID_SIMPLE_STRAT] = list(get_args(_T_VALID_SIMPLE_STRAT))
    ):

        _data_dict = {'_key': []}
        for _key, _df_slice in self.df_processed.groupby(self.group_cols):
            _df_slice = _df_slice.reset_index(drop=True)
            _data_dict['_key'].append(_key)

            strat = OHLCSimpleStrategy(_df_slice)

            buy_n_hold_res      = strat.buy_and_hold()
            buyd01_max_gain_res = strat.buy_at_first_day(result_type='gain')
            buyd01_max_loss_res = strat.buy_at_first_day(result_type='loss')
            max_gain_res        = strat.max_gain()
            max_loss_in_gl_res  = strat.max_loss(measure_type='gl')
            max_loss_in_rtn_res = strat.max_loss(measure_type='rtn')

            for res in [
                    buy_n_hold_res,
                    buyd01_max_gain_res, buyd01_max_loss_res,
                    max_gain_res, max_loss_in_gl_res, max_loss_in_rtn_res]:
                for col, val in res.to_dict().items():
                    _data_dict.setdefault(col, [])
                    _data_dict[col].append(val)

        _df = pd.DataFrame.from_dict(_data_dict)
        self._df_grouped = self._df_grouped.merge(_df, on='_key', how='left')
        return

    def _setup_overview_plot(self):

        if len(self.group_cols) == 1:
            nrows = len(self._uniq_group_key)
            ncols = 1
            subfig_idx_map = {_key: idx for idx, _key in enumerate(self._uniq_group_key)}
        else:
            _unique_key_count = [len(self.df_processed[col].unique()) for col in self.group_cols]
            nrows = int(np.prod(np.array(_unique_key_count[:-1])))
            ncols = _unique_key_count[-1]

            assert len(self._uniq_group_key) <= nrows * ncols

            # Get the index mapping 
            _key_min = [self.df_processed[col].min() for col in self.group_cols]
            _key_max = [self.df_processed[col].max() for col in self.group_cols]

            _grid = np.array(list(map(
                np.ndarray.flatten,
                np.meshgrid(*[np.arange(_min, _max+1) for _min, _max in zip(_key_min, _key_max)])
            )))
            _key_tuple = sorted([tuple(_grid[:, col]) for col in range(_grid.shape[1])])
            subfig_idx_map = {_key: idx for idx, _key in enumerate(_key_tuple)}

        return nrows, ncols, subfig_idx_map

    def _prettify_subtitle(self, _key):

        _map = {'Year': 0, 'Month': 1, 'WeekNum': 2, 'WeekDay': 3}

        res = [""] * len(_map)

        for col, val in zip(self.group_cols, _key):
            idx = _map[col]
            if col == 'WeekNum':
                _val = f'Wk {val}'
            elif col == 'WeekDay':
                _val = {0: 'Mon', 1: 'Tue', 2: 'Wed', 3: 'Thu', 4: 'Fri', 5: 'Sat', 6: 'Sun'}[val]
            elif col == "Month":
                _val = {1: "Jan", 2: "Feb", 3: "Mar", 4: "Apr",
                        5: "May", 6: "Jun", 7: "Jul", 8: "Aug",
                        9: "Sep", 10: "Oct", 11: "Nov", 12: "Dec"}[val]
            else:
                _val = str(val)

            res[idx] = _val

        return " ".join(filter(lambda x: len(x) > 0, res))
            
    def plot_overview(self,
                      x_col: Optional[ColName] = None,
                      y_cols: list[ColName] = [Col.Close],
                      type: Literal['simple', 'candle', 'renko', 'pnf', 'ohlc', 'line', 'hnf'] = 'simple',
                      mav: Optional[int | list[int] | tuple[int, ...]] = None,
                      volume: bool = False,
                      hide_xaxis: bool = True,
                      tight_layout: bool = False
                      ):

        nrows, ncols, subfig_idx_map = self._setup_overview_plot()

        with OHLCPlotter(tick_col=self.tick_col,
                         ncols=ncols,
                         nrows=nrows,
                         tight_layout=tight_layout) as plotter:

            for _key in tqdm(self._uniq_group_key, desc='Generating plots'):

                subfig_idx = subfig_idx_map[_key]
                _df = self.processor.get_slice(
                    **{x.lower(): y for x, y in zip(self.group_cols, _key)}
                )
                # Pretty ax_title
                _title = self._prettify_subtitle(_key)
                plotter.plot_subfig(
                    subfig_idx,
                    _df,
                    x_col=x_col,
                    y_cols=y_cols,
                    type=type,
                    mav=mav,
                    volume=volume,
                    hide_xaxis=hide_xaxis,
                    # 
                    subfig_title=_title
                )
            

    def plot_slice(
            self,
            year: Optional[int] = None,
            month: Optional[int] = None,
            weekday: Optional[int] = None,
            weeknum: Optional[int] = None):
        _df = self.processor.get_slice(year=year, month=month, weekday=weekday, weeknum=weeknum)

        _df = _df.set_index(self.tick_col)
        mpf.plot(_df, volume=True, mav=(3, 6, 9))
        plt.show()


class OHLCStreakAnalyzer(_OHLCBase):

    # TODO - add sanity check to make sure the columns have streak to analysis
    pass


class OHLCVolatilityAnalyzer(_OHLCBase):

    # TODO - add sanity check to make sure the columns have streak to analysis
    pass
