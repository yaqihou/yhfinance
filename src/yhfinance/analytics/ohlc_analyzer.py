
from contextlib import contextmanager

from collections import namedtuple
from dataclasses import dataclass
from typing import Literal, get_args, Optional

from tqdm import tqdm
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from .ohlc_cols import Col, ColName
from .ohlc_data import OHLCData, OHLCDataBase
from .plotter import OHLCMpfPlotter, OHLCMultiFigurePlotter
from .dataslicer import DataSlicer


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
TradingMat = namedtuple('TradingMat', ["Buy", "Sell", "Gl", 'Rtn'])
class OHLCSimpleStrategy(OHLCDataBase):
    """Applying a series of strategy on the input df.
    """

    def __init__(self,
                 data: OHLCData,
                 trading_cooldown: int = 1, # wait for at least one day
                 *,
                 cache_trading_matrix: bool = False
                 ):
        super().__init__(data)

        # Make sure the index is starting from 0
        self._df = self._df.reset_index(drop=True)
        self.trading_cooldown: int = trading_cooldown

        # _df = self._df[[self.tick_col] + list(Col.OHLC)].set_index(self.tick_col)

        # NOTE - For the most general case, we should create a matrix
        #  lazily for each buy at price and sell at price. The 2D matrix
        #  (i, j) will represent the gain / return if buy at i sell at j
        #  hence it will be an upper triangular matrix. This is slow, but
        #  could be easily implemetned and support any "cooldown" period by
        # mask off the off diagonal row
        # Useful for debug or further analysis
        self.cache_trading_matrix = cache_trading_matrix
        self._trading_matrix_cache = {}

    # def _lazy_create_trading_matrix(self, buy_at: ColName | float, sell_at: ColName | float):
    def _create_trading_matrix(self, buy_at: ColName | float, sell_at: ColName | float):
        _key_buy = buy_at if isinstance(buy_at, (ColName, str)) else None
        _key_sell = sell_at if isinstance(sell_at, (ColName, str)) else None
        _key = (_key_buy, _key_sell, self.trading_cooldown)

        if _key in self._trading_matrix_cache:
            return self._trading_matrix_cache[_key]

        sz = len(self._df)

        # Will still create a matrix for constant just to be consistent in form
        buy_vec = np.full(sz, buy_at) if _key_buy is None else self.df[buy_at]
        sell_vec = np.full(sz, sell_at) if _key_sell is None else self.df[sell_at]

        # buy mat has the same value at the same row
        buy_mat = np.repeat( np.atleast_2d(buy_vec).T, sz, axis=1 )
        # sell mat has the same value at the same col
        sell_mat = np.repeat( np.atleast_2d(sell_vec), sz, axis=0 )

        gl_mat = sell_mat - buy_mat
        # By default tril_indices will include the main diagonal, i.e. when
        # trading_cooldown = 0, we want to include the same-day trading (at
        # the main diagonal), so that we need to take the -1 offset
        gl_mat[np.tril_indices(sz, self.trading_cooldown-1)] = np.nan
        rtn_mat = gl_mat / buy_mat * 100

        ret = TradingMat(Buy=buy_mat, Sell=sell_mat, Gl=gl_mat, Rtn=rtn_mat)
        if self.cache_trading_matrix:
            self._trading_matrix_cache[_key] = ret
        return ret

    @contextmanager
    def override_params(
            self,
            trading_cooldown: Optional[int] = None # wait for at least one day
    ):
        # Setup
        default_trading_colldown = self.trading_cooldown
        self.trading_cooldown = trading_cooldown or self.trading_cooldown

        yield self

        # Tear down
        self.trading_cooldown = default_trading_colldown

        
    def buy_and_hold(
            self,
            buy_at_col: ColName = Col.Open,
            sell_at_col: ColName = Col.Close):
        """Retrun the Gl/Rtn if buy at first day and hold until the last day
        """

        buy_at=self._df.iloc[0, :].loc[buy_at_col.name]
        sell_at=self._df.iloc[-1, :].loc[sell_at_col.name]

        _gl = sell_at - buy_at
        _rtn = _gl / buy_at * 100
        _buy_day = 0
        _sell_day = len(self._df) - 1

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

        tmat = self._create_trading_matrix(
            buy_at=self._df.iloc[0, :].loc[buy_at_col.name],
            sell_at=sell_at_col
        )
        _buy_day = 0
        if result_type == 'gain':
            # Since 
            _gl = np.nanmax(tmat.Gl[_buy_day])
            _rtn = np.nanmax(tmat.Rtn[_buy_day])
            _sell_day = np.nanargmax(tmat.Gl[_buy_day])
        else:
            _gl = np.nanmin(tmat.Gl[_buy_day])
            _rtn = np.nanmin(tmat.Rtn[_buy_day])
            _sell_day = np.nanargmin(tmat.Gl[_buy_day])

        return SimpleStratRes(
            gl=_gl,
            rtn=_rtn,
            buy_day=_buy_day,
            sell_day=_sell_day,
            name = f'BuyD01{result_type.capitalize()}+{buy_at_col.name}-{sell_at_col.name}'
        )

    def max_gain(self,
                 buy_at: ColName = Col.Low,  # Could also be Close / Open
                 sell_at: ColName = Col.High,
                 ):
        """Return the max possible gain during this period, note that max
        gain could happen differently from the max return"""

        tmat = self._create_trading_matrix(
            buy_at=buy_at,
            sell_at=sell_at
        )

        _gl = np.nanmax(tmat.Gl)
        _rtn = np.nanmax(tmat.Rtn)
        _buy_day, _sell_day = np.unravel_index(np.nanargmax(tmat.Gl), tmat.Gl.shape)

        return SimpleStratRes(
            gl=_gl,
            rtn = _rtn,
            buy_day = _buy_day,
            sell_day = _sell_day,
            name = f'MaxGain+{buy_at.name}-{sell_at.name}'
        )

    def max_loss(self,
                 buy_at: ColName = Col.High,
                 sell_at: ColName = Col.Low,
                 # The min rtn could be different from min gl as the ref buy price is rollingMax
                 measure_type: Literal['gl', 'rtn'] = 'gl',
                 ):
        """Return the max possible loss during this period"""

        tmat = self._create_trading_matrix(
            buy_at=buy_at,
            sell_at=sell_at
        )

        if measure_type == 'gl':
            _buy_day, _sell_day = np.unravel_index(np.nanargmin(tmat.Gl), tmat.Gl.shape)
        elif measure_type == 'rtn':
            _buy_day, _sell_day = np.unravel_index(np.nanargmin(tmat.Rtn), tmat.Rtn.shape)
        _gl = tmat.Gl[_buy_day, _sell_day]
        _rtn = tmat.Rtn[_buy_day, _sell_day]

        return SimpleStratRes(
            gl=_gl,
            rtn = _rtn,
            buy_day = _buy_day,
            sell_day = _sell_day,
            name = f'MaxLoss{measure_type.capitalize()}+{buy_at.name}-{sell_at.name}'
        )


class OHLCSeasonalityAnalyzer(OHLCDataBase):

    def __init__(
            self,
            data: OHLCData,
            group_cols: list[DataSlicer._T_CAL_INFO_COLS] = ['Year', 'Month']
    ):
        super().__init__(data)
        self.slicer = DataSlicer(data)

        self.group_cols = group_cols

        # Prepare the summary table
        # _summary_dict = {k: [] for k in group_cols}
        self._uniq_group_key = sorted(self._df_calendar[group_cols].apply(tuple, axis=1).unique())
        # for x in self._uniq_summary_key:
        #     for k, v in zip(group_cols, x):
        #         _summary_dict[k].append(v)
        self._df_grouped = pd.DataFrame.from_dict({'_key': self._uniq_group_key})

        # self.df_slice_dict = {
        #     key: self.processor.get_slice(**{x.lower(): y for x, y in zip(group_cols, key)})
        #     for key in self._uniq_summary_key
        # }

    @property
    def df_with_cal_info(self):
        return self.slicer.df_with_calendar_info

    @property
    def _df_calendar(self):
        return self.slicer._df_calendar

    @property
    def df_grouped(self) -> pd.DataFrame:
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
        for _key, _df_slice in self.df_with_cal_info.groupby(self.group_cols):
            _df_slice = _df_slice.reset_index(drop=True)
            _data_dict['_key'].append(_key)

            _data = self._data.copy()
            _data.override_with(_df_slice)

            strat = OHLCSimpleStrategy(_data)

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

    def pivot(self, values, index, columns, **kwargs):
        """A wrapper around pd.pivot() ao apply on df_grouped"""
        return pd.pivot_table(self.df_grouped, values=values, index=index, columns=columns, **kwargs)

    def _setup_overview_plot(self):

        if len(self.group_cols) == 1:
            nrows = len(self._uniq_group_key)
            ncols = 1
            subfig_idx_map = {_key: idx for idx, _key in enumerate(self._uniq_group_key)}
        else:
            _unique_key_count = [len(self._df_calendar[col].unique()) for col in self.group_cols]
            nrows = int(np.prod(np.array(_unique_key_count[:-1])))
            ncols = _unique_key_count[-1]

            assert len(self._uniq_group_key) <= nrows * ncols

            # Get the index mapping 
            _key_min = [self._df_calendar[col].min() for col in self.group_cols]
            _key_max = [self._df_calendar[col].max() for col in self.group_cols]

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
                      type: OHLCMultiFigurePlotter._T_TYPE_LITERAL = 'line',
                      volume: bool = True,
                      mav=(3,),
                      tight_layout: bool = False,
                      **kwargs,
                      ):

        nrows, ncols, subfig_idx_map = self._setup_overview_plot()

        with OHLCMultiFigurePlotter(
                tick_col=self.tick_col,
                ncols=ncols,
                nrows=nrows,
                volume=volume, 
                tight_layout=tight_layout) as plotter:

            for _key in tqdm(self._uniq_group_key, desc='Generating plots'):

                subfig_idx = subfig_idx_map[_key]
                _df = self.slicer.get_slice(
                    **{x.lower(): y for x, y in zip(self.group_cols, _key)}
                ).set_index(self.tick_col)
                # Pretty ax_title
                _title = self._prettify_subtitle(_key)

                plotter.select_subfig(subfig_idx)
                plotter.set_subfig_suptitle(_title)
                plotter.plot_basic(_df, type=type, mav=mav, **kwargs)

    def plot_slice(
            self,
            year: Optional[int] = None,
            month: Optional[int] = None,
            qtr: Optional[int] = None,
            weekday: Optional[int] = None,
            weeknum: Optional[int] = None
    ):
        _df = self.slicer.get_slice(
            year=year, month=month, qtr=qtr, weekday=weekday, weeknum=weeknum)

        with OHLCMpfPlotter(
                tick_col=self.tick_col,
                volume=True,
        ) as plotter:
            plotter.plot_basic(_df)
        plt.show()


class OHLCStreakAnalyzer(OHLCDataBase):

    # TODO - add sanity check to make sure the columns have streak to analysis
    pass


class OHLCVolatilityAnalyzer(OHLCDataBase):

    # TODO - add sanity check to make sure the columns have streak to analysis
    pass
