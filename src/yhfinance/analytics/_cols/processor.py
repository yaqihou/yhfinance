
from collections import namedtuple

from .base import ColName, ColBase

_T_STRIKE = namedtuple("STRIKE", ['Gain', 'Loss', 'Curr'])
def _get_strike_col(base_name: str):
    return _T_STRIKE(Gain=base_name + 'GainStreak',
                     Loss=base_name + 'LossStreak',
                     Curr=base_name + "CurrStreak")

class ColIntra:
    # Intra-tick features
    Return       = ColName('IntraOpenClose')
    Swing        = ColName('IntraSwing')
    SwingGlRatio = ColName('GlSwingRatio')
    Streak       = _get_strike_col('Intra')


class ColToDay:

    Year   = ColName('YTD')
    Qtr   = ColName('QTD')
    Month = ColName('MTD')
    Week = ColName('WTD')

class ColInter:
    # Inter-tick features
    CloseOpenGap = ColName('InterGap')
    OpenCloseReturn = ColName('InterOpenClose')
    
    CloseReturn     = ColName('InterClose')
    CloseStreak     = _get_strike_col('InterClose')

    OpenReturn      = ColName('InterOpen')
    OpenStreak     = _get_strike_col('InterOpen')

    MedianReturn      = ColName('InterMedian')
    MedianStreak     = _get_strike_col('InterMedian')

    TypicalReturn      = ColName('InterTypical')
    TypicalStreak     = _get_strike_col('InterTypical')

    AvgReturn      = ColName('InterAvg')
    AvgStreak     = _get_strike_col('InterAvg')

    @classmethod
    def get_return_col(cls, price_col: ColName):
        if price_col == ColBase.Close:
            _col_res = cls.CloseReturn
        elif price_col == ColBase.Open:
            _col_res = cls.OpenReturn
        elif price_col == ColBase.Median:
            _col_res = cls.MedianReturn
        elif price_col == ColBase.Typical:
            _col_res = cls.TypicalReturn
        elif price_col == ColBase.Avg:
            _col_res = cls.AvgReturn
        else:
            raise ValueError(f"There is no corresponding return col for {price_col}")

        return _col_res


class ColRolling:
    # Rolling Window
    Year = ColName('Rolling52Wk')
    Qtr= ColName('RollingQtr')
    Month= ColName('RollingMth')
    Week = ColName('RollingWk')
