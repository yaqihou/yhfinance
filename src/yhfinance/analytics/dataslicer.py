
from copy import deepcopy
from functools import wraps
from typing import Literal, Union, Optional, get_args

import datetime as dt

import numpy as np
import pandas as pd

from .ohlc_data import OHLCDataBase, OHLCData


# TODO - need to revisit the design here to make sure it is robust
class DataSlicer(OHLCDataBase):
    """Collection of analytics for fixed time window, i.e. given year / month / week,
    suitable for seasonality analysis
    """

    _T_CAL_INFO_COLS = Literal['Year', 'Month', 'Qtr', 'WeekNum', 'WeekDay']
    _calendar_info_cols = get_args(_T_CAL_INFO_COLS)

    def __init__(
            self,
            data    : OHLCData,
            year    : Optional[int] = None,
            month   : Optional[int] = None,
            qtr     : Optional[int] = None,
            weekday : Optional[int] = None,
            weeknum : Optional[int] = None,
    ):
        super().__init__(data)
        self._parse_calendar_info()

        self._preset_year     = year
        self._preset_month    = month
        self._preset_qtr      = qtr
        self._preset_weekday  = weekday
        self._preset_weeknum  = weeknum

    def __call__(
            self,
            obj: OHLCDataBase | pd.DataFrame,
            inplace: bool = False
    ) -> Optional[pd.DataFrame | OHLCDataBase]:

        if isinstance(obj, pd.DataFrame):
            _trg_df = obj
        elif isinstance(obj, OHLCDataBase):
            _trg_df = obj._df
        else:
            _trg_df = obj._df
            print(
                'Slice is supposed to be applied to a OHLC instance but given'
                f' instance is {obj.__class__.__name__}  may work at long as obj'
                ' has an attribute _df as a PandasFrame following the same step')

        df_slice = self.get_slice(
            df      = _trg_df              ,
            year    = self._preset_year    ,
            month   = self._preset_month   ,
            qtr     = self._preset_qtr     ,
            weekday = self._preset_weekday ,
            weeknum = self._preset_weeknum ,
            copy    = True
        )

        if isinstance(obj, pd.DataFrame):
            return df_slice
        elif isinstance(obj, OHLCDataBase):
            if inplace:
                obj._df = df_slice
            else:
                new_obj = deepcopy(obj)
                new_obj._df = df_slice
                return new_obj

    def _parse_calendar_info(self):

        _df = self._df[[self.tick_col]].copy()
        # .dt can only be used with datetime but we may have dt.date type
        _df['Year'] = _df[self.tick_col].apply(lambda x: x.date().year)
        _df['Month'] = _df[self.tick_col].apply(lambda x: x.date().month)
        _df['Qtr'] = _df[self.tick_col].apply(lambda x: 1 + (x.date().month - 1) // 3)
        # For week num 53, merge it to 52
        _df['WeekNum'] = _df[self.tick_col].apply(lambda x: max(x.date().isocalendar().week, 52))
        _df['WeekDay'] = _df[self.tick_col].apply(lambda x: x.date().weekday())

        # TODO - if for intraday, we could add intraday slice

        for col in self._calendar_info_cols:
            _df[col] = _df[col].astype(int)

        self._df_calendar = _df.copy()

    @property
    def df_with_calendar_info(self):
        common_cols = set(self.df.columns) & set(self._df_calendar.columns)
        common_cols.remove(self.tick_col)
        if common_cols:
            raise ValueError(f"Some calendar info cols already existed in df, please check: {common_cols}")
        return self.df.merge(self._df_calendar, on=self.tick_col, how='left')

    def get_slice(
            self,
            df: Optional[pd.DataFrame] = None, 
            year: Optional[int] = None,
            month: Optional[int] = None,
            qtr: Optional[int] = None,
            weekday: Optional[int] = None,
            weeknum: Optional[int] = None,
            copy: bool = True,
            # ignore_calendar_info: bool = True
    ):
        """Return (a copy of) the slice from df falling in the given time window 
        """
        mask = self._df_calendar['Year'] > 0

        for col, _in in [
                ('Year', year),
                ('Month', month),
                ('Qtr', qtr),
                ('WeekNum', weeknum),
                ('WeekDay', weekday)
        ]:
            if _in is not None:
                mask = mask & (self._df_calendar[col] == _in)

        # cols = self._df.columns
        # if ignore_calendar_info:
        #     cols = list(filter(lambda x: x not in set(self.calendar_info_cols), cols))

        selected_index = self._df_calendar[mask].index

        df_target = df if df is not None else self._df
        ret = df_target.loc[selected_index, :].reset_index(drop=True)
        return ret.copy() if copy else ret

    @property
    def calendar_info_cols(self):
        return self._calendar_info_cols

    # @property
    # def year_min(self):
    #     return self.df['Year'].min()

    # @property
    # def year_max(self):
    #     return self.df['Year'].max()

    # @property
    # def years(self) -> list[int]:
    #     return sorted(self._df['Year'].unique().tolist())
