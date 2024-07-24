"""Defined a class of functions to prepare features based on OHLC data
"""

from functools import wraps
from typing import Literal, Union, Optional, get_args

import datetime as dt

import numpy as np
import pandas as pd
from pandas.core.window import Rolling

from ._base import _OHLCBase
from .const import Col, ColIntra, ColName
from . import utils

# TODO - add support to log_return

# Utility to makesure the processor input and output are correctly checked
def rectify(
        inter_tick_check: bool = False,
        # TODO
        # ensure_intraday: bool = False
        # ensure_day: bool = False
):
    def decorator(func):

        @wraps(func)
        def wrapped(self, **kwargs):

            col_res = kwargs.get('col_res')

            # Rectify the input type for ColName
            if col_res is not None:

                print('col_res in wrapped: ', col_res)

                if isinstance(col_res, str):
                    _col_res = ColName(col_res)
                elif isinstance(col_res, ColName):
                    _col_res = col_res
                else:
                    raise ValueError('col_res should be a str or ColName instance')

                kwargs['col_res'] = _col_res

            if inter_tick_check:
                if len(self._df) != len(self._df_offset):
                    self._df = self._df.merge(self._df_offset[[self.tick_col]], how='inner')

            ret = func(self, **kwargs)

            return ret

        return wrapped

    return decorator


class _OHLCBaseProcessor(_OHLCBase):

    def get_result(self):
        return self.df

    def _add_gl_streak(
            self,
            values: pd.Series,
            col_gain_streak: str,
            col_loss_streak: str,
            col_streak: str,
    ):
        """Based on the given values, add the streak columns (consecutive positive / negative)
        """

        # Count tie as gain (no loss after all)
        flags = values >= 0
        
        gain_streak = np.zeros((len(flags),), dtype=int)
        loss_streak = np.zeros((len(flags),), dtype=int)

        if flags[0]:
            gain_streak[0] += 1
        else:
            loss_streak[0] += 1

        for idx, flg in enumerate(flags[1:], 1):

            if flg:
                gain_streak[idx] = gain_streak[idx-1] + 1
            else:
                loss_streak[idx] = loss_streak[idx-1] + 1

        mask = values > 0
        self._df[col_gain_streak] = gain_streak
        self._df[col_loss_streak] = loss_streak
        self._df[col_streak] = self._df[col_loss_streak]
        self._df.loc[mask, col_streak] = self._df[col_gain_streak]

        return self

    def __add__(self, obj: _OHLCBase):

        common_cols = list(set(self._df.columns) & set(obj._df.columns))
        _df = self._df.merge(obj._df, on=common_cols, how='left')

        return _OHLCBaseProcessor(df=_df, tick_col=self.tick_col)



class OHLCIntraProcessor(_OHLCBaseProcessor):
    """Populate intra-tick features
    """

    def add_all(self, add_rtn_col: bool = True):
        """A shortcut to apply all additional features, not support customize result col name
        """
        return self.add_swing(add_rtn_col=add_rtn_col)\
                   .add_gl(add_rtn_col=add_rtn_col)\
                   .add_gl_swing_ratio()\
                   .add_gl_streak()

    @rectify()
    def add_swing(self, col_res: ColName = Col.Intra.Swing, add_rtn_col: bool = True):
        """Return the difference (High - Low) if Close >= Open, (Low - High) if Close < Open
        """
        self._df[col_res.name] = self._df[Col.High.name] - self._df[Col.Low.name]
        _mask = self._df[Col.Close.name] < self._df[Col.Open.name]
        self._df.loc[_mask, col_res.name] *= -1

        if add_rtn_col:
            self._df[col_res.rtn] = self._df[col_res.name] / self._df[Col.Low.name] * 100
            self._df[col_res.ln_rtn] = np.log(self._df[Col.High.name] / self._df[Col.Low.name]) * 100

        return self

    @rectify()
    def add_gl(self, col_res: ColName = Col.Intra.Return, add_rtn_col: bool = True):
        """Return the intra tick difference (Close - Open)
        """

        self._df[col_res.gl] = self._df[Col.Close.name] - self._df[Col.Open.name]
        if add_rtn_col:
            self._df[col_res.rtn] = self._df[col_res.gl] / self._df[Col.Open.name] * 100
            self._df[col_res.ln_rtn] = np.log(self._df[Col.Close.name] / self._df[Col.Open.name]) * 100

        return self

    @rectify()
    def add_gl_swing_ratio(self, col_res: ColName = Col.Intra.SwingGlRatio):
        """Return the ratio of return / spread, i.e. (Close - Open) / (High - Low), share the same sign
        as swing
        """

        self._df[col_res.name] = (
            (self._df[Col.Close.name] - self._df[Col.Open.name])
            / (self._df[Col.High.name] - self._df[Col.Low.name])
        )

        return self

    def add_gl_streak(self):
        """Get the gain or lose streak, i.e. X ticks in row we see gain / lose in (Close - Open)
        """
        values = self._df[Col.Close.name] - self._df[Col.Open.name]
        self._add_gl_streak(
            values, Col.Intra.GainStreak.name, Col.Intra.LossStreak.name, Col.Intra.Streak.name
        )
        return self


class OHLCInterProcessor(_OHLCBaseProcessor):

    def __init__(
            self, df: pd.DataFrame,
            tick_col: str = Col.Date.name,
            tick_offset: int = -1):
        """time_offset: used when joining by tick difference"""
        super().__init__(df, tick_col)

        self.tick_offset = tick_offset
        
        self._df_offset = self._join_by_time_diff(
            self._df,
            tick_offset=tick_offset,
            tick_col=tick_col,
            copy = True
        )

    @staticmethod
    def _join_by_time_diff(
            df: pd.DataFrame,
            tick_offset: int = -1,
            tick_col: str = Col.Date.name,
            suffixes: tuple[str, str] = ColName.suffixes,
            value_columns: list[str] | tuple[str] = Col.All,
            copy: bool = True
    ):
        """Join based on the key_col, with shifted row
        """

        if copy is True:  df = df.copy()

        df = df.sort_values(by=tick_col)
        # tmp col for the target date
        if tick_offset > 0:
            df['ShiftedTick'] = df[tick_col].values.tolist()[tick_offset:] + [pd.NaT] * tick_offset
        else: 
            df['ShiftedTick'] = [pd.NaT] * abs(tick_offset) + df[tick_col].values.tolist()[:tick_offset]
        df['ShiftedTick'] = pd.to_datetime(df['ShiftedTick'])

        df = pd.merge(df, df[[tick_col, *value_columns]].rename(columns={tick_col: 'ShiftedTick'}),
                      on='ShiftedTick', how='inner', suffixes=suffixes)
        # df.drop(columns='ShiftedTime', inplace=True)
        return df

    def _add_gl_return(self, col_res: ColName, col_a_name: str, col_b_name: str, add_rtn_col: bool):
        """Add the column col_ref for [col_a_name] - [col_b_name]
        """

        self._df[col_res.gl] = self._df_offset[col_a_name] - self._df_offset[col_b_name]
        if add_rtn_col:
            self._df[col_res.rtn] = self._df[col_res.gl] / self._df_offset[col_b_name] * 100
            self._df[col_res.ln_rtn] = np.log(self._df_offset[col_a_name] / self._df_offset[col_b_name]) * 100

        return self

    def add_all(self, add_rtn_col: bool = True):
        """A shortcut to apply all additional features, not support customize result col name
        """

        return self.add_close_open_spread(add_rtn_col=add_rtn_col)\
                   .add_close_return(add_rtn_col=add_rtn_col)\
                   .add_open_return(add_rtn_col=add_rtn_col)\
                   .add_open_close_return(add_rtn_col=add_rtn_col)\
                   .add_close_gl_streak()\
                   .add_open_gl_streak()

    @rectify(inter_tick_check=True)
    def add_close_open_spread(self, *, col_res=Col.Inter.CloseOpenSpread, add_rtn_col: bool = True):
        """Add the column for the spread of Open_t - Close_{t-1}
        """

        self._add_gl_return(col_res, Col.Open.cur, Col.Close.sft, add_rtn_col)
        return self

    @rectify(inter_tick_check=True)
    def add_open_return(self, *, col_res=Col.Inter.OpenReturn, add_rtn_col: bool = True):
        """Add the column for the spread of Open_t - open_{t-1} """

        self._add_gl_return(col_res, Col.Open.cur, Col.Open.sft, add_rtn_col)
        return self

    @rectify(inter_tick_check=True)
    def add_close_return(self, *, col_res=Col.Inter.CloseReturn, add_rtn_col: bool = True):
        """Add the column for the spread of Close_t - Close_{t-1} """

        # Close_T - Close{T-1}
        self._add_gl_return(col_res, Col.Close.cur, Col.Close.sft, add_rtn_col)
        return self

    @rectify(inter_tick_check=True)
    def add_open_close_return(self, *, col_res=Col.Inter.OpenCloseReturn, add_rtn_col: bool = True):
        """Add the column for the spread of Close_t - open_{t-1} """

        self._add_gl_return(col_res, Col.Close.cur, Col.Open.sft, add_rtn_col)
        return self

    @rectify(inter_tick_check=True)
    def add_close_gl_streak(self):
        """Get the gain or lose streak based on inter tick change (Close_T - Close_{T-1})
        """

        values = self._df_offset[Col.Close.cur] - self._df_offset[Col.Close.sft]
        self._add_gl_streak(
            values,
            Col.Inter.CloseGainStreak.name,
            Col.Inter.CloseLossStreak.name,
            Col.Inter.CloseStreak.name
        )
        return self
    
    @rectify(inter_tick_check=True)
    def add_open_gl_streak(self):
        """Get the gain or lose streak based on inter tick change (Open_T - Open_{T-1})
        """

        values = self._df_offset[Col.Open.cur] - self._df_offset[Col.Open.sft]
        self._add_gl_streak(
            values,
            Col.Inter.OpenGainStreak.name,
            Col.Inter.OpenLossStreak.name,
            Col.Inter.OpenStreak.name
        )
        return self


_T_WND_SIZE = Union[int, str, dt.timedelta, ]
_T_VOL_TYPE = Literal['Intra', 'InterClose', 'InterOpen', 'Parkinson', 'Garman-Klass', 'GK']
_T_RTN_TYPE = Literal['simple', 'log', 'ln']
class OHLCTrailingProcessor(_OHLCBaseProcessor):

    def __init__(self,
                 df: pd.DataFrame,
                 tick_col: str = Col.Date.name,
                 padding: Literal['keep', 'drop'] = 'keep'):
        """
        padding [bool]: keep or drop for the first n observations in the rolling window
        """
        super().__init__(df, tick_col)

        self.padding = padding
        if padding != 'keep':
            raise NotImplementedError("padding method support has not been implemented yet")

    # ---------------------------
    # Add high_low
    def _add_trailing_high_low(self, window_size: _T_WND_SIZE, col_res: ColName,
            allow_partial_window_values: bool = True):
        """Add the high and low in the trailing window size """

        # NOTE - below is an old implementation based on heap, which might be useful if we
        #        want more complicated intermedaite information dumped. But for now, then
        #        .rolling() method is easier to implement
        # def populate_window_high_n_low(df, window_size=52):
        # 
        #     data = zip(df['Date'], df['Close'])
        #     pq_high = []
        #     pq_low = []
        # 
        #     hist_highs = []
        #     hist_high_dates = []
        # 
        #     hist_lows = []
        #     hist_low_dates = []
        # 
        #     for date, close in data:
        #         # Push into heap at first as current date could be the new high/low
        #         heapq.heappush(pq_high, (-close, date)) # Note the sign
        #         heapq.heappush(pq_low, (close, date))
        # useful        
        #         while pq_high and date - pq_high[0][1] > dt.timedelta(weeks=window_size):
        #             heapq.heappop(pq_high)
        # 
        #         hist_highs.append(-pq_high[0][0])
        #         hist_high_dates.append(pq_high[0][1])
        # 
        #         # --------------------------- 
        #         while pq_low and date - pq_low[0][1] > dt.timedelta(weeks=window_size):
        #             heapq.heappop(pq_low)
        # 
        #         hist_lows.append(pq_low[0][0])
        #         hist_low_dates.append(pq_low[0][1])
        # 
        # 
        #     df[f'{window_size} Wk High'] = hist_highs
        #     df[f'{window_size} Wk High Date'] = hist_high_dates
        # 
        #     df[f'{window_size} Wk Low'] = hist_lows
        #     df[f'{window_size} Wk Low Date'] = hist_low_dates
        # 
        #     return df
        
        _min_periods = 1 if allow_partial_window_values else None
        
        _trailing_high = self._df[Col.High.name].rolling(window_size, min_periods=_min_periods).max()
        _trailing_low = self._df[Col.Low.name].rolling(window_size, min_periods=_min_periods).min()

        self._df[col_res.high] = _trailing_high
        self._df[col_res.low] = _trailing_low

        return self

    def add_trailing_year_high_low(self, window_size: _T_WND_SIZE=252,
                                  allow_partial_window_values: bool = True):
        """Add the high and low in the trailing year (252 ticks)
        """
        self._add_trailing_high_low(
            window_size, Col.Rolling.Year,
            allow_partial_window_values=allow_partial_window_values)
        return self
        
    def add_trailing_qtr_high_low(self, window_size: _T_WND_SIZE=63,
                                  allow_partial_window_values: bool = True):
        """Add the high and low in the trailing quarter (63 ticks)
        """
        self._add_trailing_high_low(
            window_size, Col.Rolling.Qtr,
            allow_partial_window_values=allow_partial_window_values)
        return self

    def add_trailing_mth_high_low(self, window_size: _T_WND_SIZE=21,
                                  allow_partial_window_values: bool = True):
        """Add the high and low in the trailing month (21 ticks)
        """
        self._add_trailing_high_low(
            window_size, Col.Rolling.Month,
            allow_partial_window_values=allow_partial_window_values)
        return self

    def add_trailing_wk_high_low(self, window_size: _T_WND_SIZE=5,
                                  allow_partial_window_values: bool = True):
        """Add the high and low in the trailing week (5 ticks)
        """
        self._add_trailing_high_low(
            window_size, Col.Rolling.Week,
            allow_partial_window_values=allow_partial_window_values)
        return self

    # ---------------------------
    # Add volatility
    def _add_trailing_hist_vol(
            self,
            window_size: _T_WND_SIZE,
            col_trailing: ColName,
            vol_type: _T_VOL_TYPE = 'Intra',  # only used when col_rtn is None
            rtn_type: _T_RTN_TYPE = 'simple',
            col_rtn_name: Optional[str] = None,
            allow_partial_window_values: bool = False):
        """Calculate the volatilities for the trailing window

        col_rtn has the highest priority and will assume the df has this column already
        if col_rtn is NOne, vol_type and rtn_type will be used to infer the formula
        For rtn_type is only used when vol_type = 'Inter' / 'Intra'
        For vol_type = 'GK' / 'Parkinson', rtn_type will be ignored
        """

        # _min_periods = 1 if allow_partial_window_values else None
        if allow_partial_window_values:
            print("Warning: Trailing hist_vol will not allow partial window")

        # Check if there are data existed

        if col_rtn_name is not None:
            # Use the given return col directly
            assert col_rtn_name in self._df.columns
            _trailing_vol = self._df[col_rtn_name].rolling(window_size).std(ddof=1)
            _col_res_name = col_trailing.name + col_rtn_name + 'Vol'

        else:

            if vol_type.startswith('Inter'):
                _processor = OHLCInterProcessor(self._df_raw.copy())
                _df_offset = _processor.add_all().get_result()

                if vol_type == 'InterClose':
                    _col = Col.Inter.CloseReturn
                elif vol_type == 'InterOpen':
                    _col = Col.Inter.OpenReturn
                else:
                    raise ValueError(f"Encounter unknown vol_type: {vol_type}")

                _vol_name = vol_type + ("Simple" if rtn_type == 'simple' else 'Ln')
                _col_res_name = col_trailing.name + _vol_name + 'Vol'

                _col_rtn_name = _col.rtn if rtn_type == 'simple' else _col.ln_rtn
                _df_offset[_col_res_name] = _df_offset[_col_rtn_name].rolling(window_size).std(ddof=1)

                # Note that _df_offset has different size from the original _df, so that we use merge
                self._df = self._df.merge(
                    _df_offset[[self.tick_col, _col_res_name]], how='left', on=self.tick_col
                )

            else:
                # Inter vol
                _processor = OHLCIntraProcessor(self._df_raw.copy())
                _df = _processor.add_all().get_result()
                if vol_type == 'Intra':
                    _col_rtn_name = Col.Intra.Return.rtn if rtn_type == 'simple' else Col.Intra.Return.ln_rtn
                    _trailing_vol = _df[_col_rtn_name].rolling(window_size).std(ddof=1)
                    _vol_name = vol_type + ("Simple" if rtn_type == 'simple' else 'Ln')
                elif vol_type == 'Parkinson':
                    _df['TmpRtn'] = _df[Col.Intra.Swing.ln_rtn] ** 2
                    _trailing_vol = _df['TmpRtn'].rolling(window_size).mean()
                    _trailing_vol = np.sqrt(_trailing_vol / (4 * np.log(2)))
                    _vol_name = 'Pksn'
                elif vol_type == 'Garman-Klass' or vol_type == 'GK':
                    _df['TmpRtn'] = (
                        0.5 * (np.log( _df[Col.High.name] / _df[Col.Low.name] )) ** 2
                        - (2 * np.log(2) - 1) * (np.log( _df[Col.Close.name] / _df[Col.Open.name] )) ** 2
                        )
                    # Since we scale all return / log_return into pct
                    _trailing_vol = np.sqrt(_df['TmpRtn'].rolling(window_size).mean()) * 100
                    _vol_name = 'GK'
                else:
                    raise ValueError(f"Encounter unknown vol_type: {vol_type}")

                _col_res_name = col_trailing.name + _vol_name + 'Vol'
                self._df[_col_res_name] = _trailing_vol
    
        return self

    def add_trailing_year_volatility(
            self, 
            vol_type: _T_VOL_TYPE = 'Intra',  # only used when col_rtn is None
            rtn_type: _T_RTN_TYPE = 'simple',
            col_rtn_name: Optional[str] = None,
            *,
            window_size: _T_WND_SIZE = 252
    ):

        self._add_trailing_hist_vol(
                window_size = window_size,
                col_trailing = Col.Rolling.Year,
                vol_type=vol_type,  # only used when col_rtn_name is None
                rtn_type=rtn_type,
                col_rtn_name=col_rtn_name)
        return self

    def add_trailing_qtr_volatility(
            self, 
            vol_type: _T_VOL_TYPE = 'Intra',  # only used when col_rtn_name is None
            rtn_type: _T_RTN_TYPE = 'simple',
            col_rtn_name: Optional[str] = None,
            *,
            window_size: _T_WND_SIZE = 63
    ):
        self._add_trailing_hist_vol(
                window_size = window_size,
                col_trailing = Col.Rolling.Qtr,
                vol_type=vol_type,  # only used when col_rtn_name is None
                rtn_type=rtn_type,
                col_rtn_name=col_rtn_name)
        return self

    def add_trailing_mth_volatility(
            self, 
            vol_type: _T_VOL_TYPE = 'Intra',  # only used when col_rtn_name is None
            rtn_type: _T_RTN_TYPE = 'simple',
            col_rtn_name: Optional[str] = None,
            *,
            window_size: _T_WND_SIZE = 21
    ):
        self._add_trailing_hist_vol(
                window_size = window_size,
                col_trailing = Col.Rolling.Month,
                vol_type=vol_type,  # only used when col_rtn_name is None
                rtn_type=rtn_type,
                col_rtn_name=col_rtn_name)
        return self

    def add_trailing_wk_volatility(
            self, 
            vol_type: _T_VOL_TYPE = 'Intra',  # only used when col_rtn_name is None
            rtn_type: _T_RTN_TYPE = 'simple',
            col_rtn_name: Optional[str] = None,
            *,
            window_size: _T_WND_SIZE = 5
    ):
        self._add_trailing_hist_vol(
                window_size = window_size,
                col_trailing = Col.Rolling.Week,
                vol_type=vol_type,  # only used when col_rtn_name is None
                rtn_type=rtn_type,
                col_rtn_name=col_rtn_name)
        return self


# TODO - move all indicators outside processor into its own class
#        each indicator will have its own class
#        this way, we can define ploting function inside
class OHLCIndicatorProcessor(_OHLCBaseProcessor):

    def add_macd(self,
                 short_term_window: int = 12,
                 long_term_window: int = 26,
                 signal_window: int = 9,
                 price_col: ColName = Col.Close):

        ewm_short = self._df[price_col.name].ewm(span=short_term_window, adjust=False).mean()
        ewm_long = self._df[price_col.name].ewm(span=long_term_window, adjust=False).mean()

        macd = ewm_short - ewm_long
        signal = macd.ewm(span=signal_window, adjust=False).mean()
        
        self._df[Col.Ind.Momentum.MACD.EMA12.name] = ewm_short
        self._df[Col.Ind.Momentum.MACD.EMA26.name] = ewm_long
        self._df[Col.Ind.Momentum.MACD.MACD.name] = macd
        self._df[Col.Ind.Momentum.MACD.Signal.name] = signal

        return self

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

    def _get_gl_for_rsi(self):

        _df = OHLCInterProcessor(self.df, tick_offset=-1).add_close_return().get_result()
        _df = _df[[self.tick_col, Col.Inter.CloseReturn.gl]].copy()

        ups = _df[Col.Inter.CloseReturn.gl].apply(lambda x: max(x, 0))
        dns = _df[Col.Inter.CloseReturn.gl].apply(lambda x: -min(x, 0))
        _df = _df.drop(columns=[Col.Inter.CloseReturn.gl])

        return _df, ups, dns

    def _assign_rsi_result(self, df, col_rsi, n, ewm_ups, ewm_dns):
        
        df[col_rsi.AvgGain(n)] = ewm_ups
        df[col_rsi.AvgLoss(n)] = ewm_dns
        df[col_rsi.RS(n)] = ewm_ups / ewm_dns
        df[col_rsi.RSI(n)] = 100 - (100 / (1 + ewm_ups / ewm_dns))

        self._add_result_safely(df)

        return

    def add_rsi_wilder(self, n: int = 14):
        """Using the original formula from Wilder
        U / D is the difference between close prices,
        using SMMA with alpha = 1 / n, i.e. y_t = (1-a) * y_t-1 + a * x_t 
        First n -1 data is set to nan, and initial result starts from nth element
        as the 
        """
        _df, ups, dns = self._get_gl_for_rsi()

        ups.iloc[n] = ups.iloc[:n].mean()
        ups.iloc[:n] = pd.NA
        dns.iloc[n] = dns.iloc[:n].mean()
        dns.iloc[:n] = pd.NA

        alpha = 1 / n
        ewm_ups = ups.ewm(alpha=alpha, ignore_na=True, adjust=False).mean()
        ewm_dns = dns.ewm(alpha=alpha, ignore_na=True, adjust=False).mean()

        self._assign_rsi_result(_df, Col.Ind.Momentum.RSIWilder, n, ewm_ups, ewm_dns)

        return self

    def add_rsi_ema(self, n: int = 14):
        """The main difference from wilder's original is that instead of SMMA, we used a EMA
        so that the first n obs are NOT nan
        o"""
        _df, ups, dns = self._get_gl_for_rsi()

        alpha = 1 / n
        ewm_ups = ups.ewm(alpha=alpha, adjust=False).mean()
        ewm_dns = dns.ewm(alpha=alpha, adjust=False).mean()

        self._assign_rsi_result(_df, Col.Ind.Momentum.RSIEma, n, ewm_ups, ewm_dns)

        return self

    def add_rsi_cutler(self, n: int = 14):
        """Cutler's RSI variation is based on SMA, to overcome the so-called 'Data Length Dependency'"""
        _df, ups, dns = self._get_gl_for_rsi()

        ewm_ups = ups.rolling(n).mean()
        ewm_dns = dns.rolling(n).mean()

        self._assign_rsi_result(_df, Col.Ind.Momentum.RSICutler, n, ewm_ups, ewm_dns)

        return self

    def add_supertrend(self,
                       period: int = 7,
                       multiplier: int = 3,
                       multiplier_dn: Optional[int] = None  # if None, the same as multiplier
                       ):
        """Add calculations related to Supertrend up/dn and the actual to display

        The best thing about supertrend it sends out accurate signals
        on precise time. The indicator is available on various trading
        platforms free of cost. The indicator offers quickest technical
        analysis to enable the intraday traders to make faster
        decisions. As said above, it is extremely simple to use and
        understand.
        
        However, the indicator is not appropriate for all the
        situations. It works when the market is trending. Hence it is best
        to use for short-term technical analysis. Supertrend uses only the
        two parameters of ATR and multiplier which are not sufficient under
        certain conditions to predict the accurate direction of the market.
        
        Understanding and identifying buying and selling signals in
        supertrend is the main crux for the intraday traders. Both the
        downtrends as well uptrends are represented by the tool. The
        flipping of the indicator over the closing price indicates
        signal. A buy signal is indicated in green color whereas sell
        signal is given as the indicator turns red. A sell signal occurs
        when it closes above the price.
        """

        _df = OHLCInterProcessor(self.df, tick_offset=-1)._df_offset
        # TR = max[(H-L), abs(H-Cp), abs(L-Cp)]
        _df[Col.Ind.Band.TrueRange.name] = pd.concat([
            _df[Col.High.cur] - _df[Col.Low.cur],
            (_df[Col.High.cur] - _df[Col.Close.sft]).abs(),
            (_df[Col.Low.cur] - _df[Col.Close.sft]).abs()
            ], axis=1).max(axis=1)

        # ATR_curr = ATR_prev * (n-1) / n + TR_curr * (1/n)
        # i.e. a EWM with alpha = 1/n
        # initial condition is ATR_0 = mean(sum(TR_n))

        _tr = _df[Col.Ind.Band.TrueRange.name].copy()
        _tr.iloc[period] = _tr.iloc[:period].mean()
        _tr.iloc[:period] = pd.NA

        alpha = 1 / period
        _df[Col.Ind.Band.AvgTrueRange(period)] = _tr.ewm(alpha=alpha, ignore_na=True, adjust=False).mean()

        _multi_up = multiplier
        _multi_dn = multiplier_dn or multiplier

        # NOTE - the rolling min is one way to use but that will introduce another paratermeter for the
        #        window size
        _df[Col.Ind.Band.SuperTrendUp(period, _multi_up)] = (
            0.5 * (_df[Col.High.cur] + _df[Col.Low.cur])
            + _multi_up * _df[Col.Ind.Band.AvgTrueRange(period)]
        )#.rolling(period).min()
        _df[Col.Ind.Band.SuperTrendDn(period, _multi_dn)] = (
            0.5 * (_df[Col.High.cur] + _df[Col.Low.cur])
            - _multi_dn * _df[Col.Ind.Band.AvgTrueRange(period)]
        )#.rolling(period).max()

        if _multi_dn == _multi_up:
            _col_supertrend_name = Col.Ind.Band.SuperTrend(period, _multi_up)
        else:
            _col_supertrend_name = Col.Ind.Band.SuperTrend(period, _multi_up, _multi_dn)

        supertrend = np.full((len(_df),), np.nan)
        base_ups = _df[Col.Ind.Band.SuperTrendUp(period, _multi_up)].to_list()
        base_dns = _df[Col.Ind.Band.SuperTrendDn(period, _multi_dn)].to_list()
        final_ups = np.full((len(_df),), np.nan)
        final_dns = np.full((len(_df),), np.nan)
        closes = _df[Col.Close.cur].to_list()

        final_ups[period] = base_ups[period]
        final_dns[period] = base_dns[period]
        supertrend[period] = base_ups[period]
        # 1 for showing support, 0 for showing resistence
        current_mode = 1
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
                current_mode = 0
            # Break resistence, switch to support mode
            elif closes[idx] > final_ups[idx]:
                current_mode = 1
            # Otherwise, just keep current trend
            supertrend[idx] = final_dns[idx] if current_mode == 1 else final_ups[idx]

        # # TrendUp is the resistence
        _df[_col_supertrend_name] = supertrend
        # TODO - put the final bands to df
        # _df[Col.Ind.Band.SuperTrendUp(period, _multi_up)] = final_ups
        # _df[Col.Ind.Band.SuperTrendDn(period, _multi_dn)] = final_dns

        _df = _df[[
            self.tick_col,
            Col.Ind.Band.TrueRange.name,
            Col.Ind.Band.AvgTrueRange(period),
            Col.Ind.Band.SuperTrendUp(period, _multi_up),
            Col.Ind.Band.SuperTrendDn(period, _multi_dn),
            _col_supertrend_name
        ]]

        self._add_result_safely(_df)
        return self

        
class OHLCToDayProcessor(_OHLCBaseProcessor):

    # TODO - could support intraday tick
    _GET_NEXT_PERIOD_START_FUNCS = {
            'year': utils.get_next_year_start,
            'qtr': utils.get_next_qtr_start,
            'mth': utils.get_next_mth_start,
            'week': utils.get_next_week_start,
        }

    @staticmethod
    def _get_period_start(
            dates,
            period_type: Literal['year', 'qtr', 'mth', 'week']):
        """Return the list of first date in dates for a given period"""

        # TODO - could support intraday tick
        func = OHLCToDayProcessor._GET_NEXT_PERIOD_START_FUNCS[period_type]

        ret = [dates[0]]
        curr_period_start = dates[0]
        next_period_start = func(curr_period_start)

        for idx, date in enumerate(dates[1:], 1):
            if date >= next_period_start:
                curr_period_start = date
                next_period_start = func(curr_period_start)

            ret.append(curr_period_start)
            
        return ret

    def _add_to_day_gl_rtn(
            self,
            period_type: Literal['year', 'qtr', 'mth', 'week'],
            col_ref: ColName = Col.Open):
        """Add the gl and rtn columns with respect to the start of the given period,
        with respect to Open / Close price"""

        col_res: ColName = {
            'year' : Col.ToDay.Year,
            'qtr'  : Col.ToDay.Qtr,
            'mth'  : Col.ToDay.Month,
            'week' : Col.ToDay.Week,
        }[period_type]


        dates = self._df[self.tick_col].dt.to_pydatetime().tolist()
        _col_start = col_res.name + 'Start'
        self._df[_col_start] = self._get_period_start(dates, period_type)

        cols = [self.tick_col, Col.Open.name, Col.Close.name]
        _df = self._df.merge(self._df[cols].rename(columns={self.tick_col: _col_start}),
                             on='PeriodStart', suffixes=ColName.suffixes)

        self._df[col_res.gl] = _df[Col.Close.cur] - _df[col_ref.sft]
        self._df[col_res.rtn] = self._df[col_res.gl] / _df[col_ref.sft] * 100
        self._df[col_res.ln_rtn] = np.log(_df[Col.Close.cur] / _df[col_ref.sft]) * 100
        
        return self

    def add_year_to_day_gl_rtn(self, col_ref: ColName = Col.Open,
                               allow_partial_window_values: bool = True):
        """Add the gl and rtn columns with respect to the start of this year"""
        self._add_to_day_gl_rtn(period_type='year', col_ref=col_ref)
        return self
        
    def add_qtr_to_day_gl_rtn(self, col_ref: ColName = Col.Open,
                              allow_partial_window_values: bool = True):
        """Add the high and low in the year to day period"""
        self._add_to_day_gl_rtn(period_type='qtr', col_ref=col_ref)
        return self

    def add_mth_to_day_gl_rtn(self, col_ref: ColName = Col.Open,
                              allow_partial_window_values: bool = True):
        """Add the high and low in the year to day period"""
        self._add_to_day_gl_rtn(period_type='mth', col_ref=col_ref)
        return self

    def add_week_to_day_gl_rtn(self, col_ref: ColName = Col.Open,
                               allow_partial_window_values: bool = True):
        """Add the high and low in the week to day period"""
        self._add_to_day_gl_rtn(period_type='week', col_ref=col_ref)
        return self

    # NOTE - It does not make much sense to calculate volatility to date, as it will
    #        introduce some noise when the period starts
    # @staticmethod
    # def _get_historical_volatility(
    #         dates, values,
    #         period_type: Literal['year', 'qtr', 'mth', 'week']
    # ):
    #     # TODO: not memory efficient
    #     func = OHLCToDayProcessor._GET_NEXT_PERIOD_START_FUNCS[period_type]

    #     tmp = [[values[0]]]
    #     next_period_start = func(dates[0])

    #     for idx, date in enumerate(dates[1:], 1):
    #         value = values[idx]
    #         if date >= next_period_start:
    #             next_period_start = func(date)
    #             tmp.append([value])
    #         else:
    #             tmp.append(tmp[-1] + [value])

    #     ret = list(map(lambda x: np.std(x), tmp))
            
    #     return ret

    # def _add_historical_volatility(
    #         self,
    #         col_rtn_name: str,
    #         col_res: ColName,
    #         period_type: Literal['year', 'qtr', 'mth', 'week'],
    #         annualized: bool = False,  # sigma_T = sigma_D * sqrt(252)
    # ):

    #     if col_rtn_name not in self._df.columns:
    #         raise ValueError(f'Given return column {col_rtn.rtn} not existed')

    #     dates = self._df[self.tick_col]
    #     values = self._df[col_rtn_name]

    #     _hist_sigma = self._get_historical_volatility(dates, values, period_type=period_type)

    #     self._df[col_res.hist_sigma] = _hist_sigma

    @staticmethod
    def _get_to_day_low(
            dates, values,
            period_type: Literal['year', 'qtr', 'mth', 'week']
    ):
        """Return the list of first date in dates for a given period"""
        func = OHLCToDayProcessor._GET_NEXT_PERIOD_START_FUNCS[period_type]

        ret = [values[0]]
        _min = values[0]
        next_period_start = func(dates[0])

        for idx, date in enumerate(dates[1:], 1):
            value = values[idx]
            if date >= next_period_start:
                next_period_start = func(date)
                ret.append(value)
                _min = value
            else:
                ret.append(min(value, _min))
            
        return ret


    @staticmethod
    def _get_to_day_high(
            dates, values,
            period_type: Literal['year', 'qtr', 'mth', 'week']
    ):
        """Return the list of first date in dates for a given period"""
        # TODO - this is not very memory friendly

        # TODO - could support intraday tick
        func = OHLCToDayProcessor._GET_NEXT_PERIOD_START_FUNCS[period_type]

        ret = [values[0]]
        _max = values[0]
        next_period_start = func(dates[0])

        for idx, date in enumerate(dates[1:], 1):
            value = values[idx]
            if date >= next_period_start:
                next_period_start = func(date)
                ret.append(value)
                _max = value
            else:
                ret.append(max(value, _max))
            
        return ret

    def _add_to_day_high_low(
            self, col_res: ColName,
            period_type: Literal['year', 'qtr', 'mth', 'week']):

        dates = self._df[self.tick_col].dt.to_pydatetime().tolist()

        values = self._df[Col.Low.name].to_list()
        _lows = self._get_to_day_low(dates, values, period_type=period_type)
        self._df[col_res.low] = _lows

        values = self._df[Col.High.name].to_list()
        _highs = self._get_to_day_high(dates, values, period_type=period_type)
        self._df[col_res.high] = _highs

        return self

    def add_year_to_day_high_low(self, allow_partial_window_values: bool = True):
        """Add the high and low in the year to day period"""
        self._add_to_day_high_low(Col.ToDay.Year, period_type='year')
        return self
        
    def add_qtr_to_day_high_low(self, allow_partial_window_values: bool = True):
        """Add the high and low in the year to day period"""
        self._add_to_day_high_low(Col.ToDay.Qtr, period_type='qtr')
        return self

    def add_mth_to_day_high_low(self, allow_partial_window_values: bool = True):
        """Add the high and low in the month to day period"""
        self._add_to_day_high_low(Col.ToDay.Month, period_type='mth')
        return self

    def add_wk_to_day_high_low(self, allow_partial_window_values: bool = True):
        """Add the high and low in the week to day period"""
        self._add_to_day_high_low(Col.ToDay.Week, period_type='week')
        return self


# TODO - Ensure day history
class OHLCFixedWindowProcessor(_OHLCBaseProcessor):
    """Collection of analytics for fixed time window, i.e. given year / month / week,
    suitable for seasonality analysis
    """

    _T_CAL_INFO_COLS = Literal['Year', 'Month', 'Qtr', 'WeekNum', 'WeekDay']
    _calendar_info_cols = get_args(_T_CAL_INFO_COLS)

    def __init__(
            self,
            df: pd.DataFrame, tick_col: str = Col.Date.name,
    ):
        super().__init__(df, tick_col)
        self._add_calendar_info()

    def _add_calendar_info(self):

        # .dt can only be used with datetime but we may have dt.date type
        self._df['Year'] = self._df[self.tick_col].apply(lambda x: x.date().year)
        self._df['Month'] = self._df[self.tick_col].apply(lambda x: x.date().month)
        self._df['Qtr'] = self._df[self.tick_col].apply(lambda x: 1 + (x.date().month - 1) // 3)
        # For week num 53, merge it to 52
        self._df['WeekNum'] = self._df[self.tick_col].apply(lambda x: max(x.date().isocalendar().week, 52))
        self._df['WeekDay'] = self._df[self.tick_col].apply(lambda x: x.date().weekday())

        for col in self._calendar_info_cols:
            self._df[col] = self._df[col].astype(int)

    def get_slice(
            self,
            year: Optional[int] = None,
            month: Optional[int] = None,
            qtr: Optional[int] = None,
            weekday: Optional[int] = None,
            weeknum: Optional[int] = None,
            copy: bool = True,
            ignore_calendar_info: bool = True
    ):
        """Return (a copy of) the slice from df falling in the given time window 
        """
        mask = self._df['Year'] > 0

        for col, _in in [
                ('Year', year),
                ('Month', month),
                ('Qtr', qtr),
                ('WeekNum', weeknum),
                ('WeekDay', weekday)
        ]:
            if _in is not None:
                mask = mask & (self._df[col] == _in)

        cols = self._df.columns
        if ignore_calendar_info:
            cols = list(filter(lambda x: x not in set(self.calendar_info_cols), cols))
        ret = self._df.loc[mask, cols].reset_index(drop=True)
        return ret.copy() if copy else ret

    @property
    def calendar_info_cols(self):
        return self._calendar_info_cols

    @property
    def year_min(self):
        return self._df['Year'].min()

    @property
    def year_max(self):
        return self._df['Year'].max()

    @property
    def years(self) -> list[int]:
        return sorted(self._df['Year'].unique().tolist())
