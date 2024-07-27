
from collections import namedtuple

from .base import ColName

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
    CloseOpenSpread = ColName('InterSpread')
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


class ColRolling:
    # Rolling Window
    Year = ColName('Rolling52Wk')
    Qtr= ColName('RollingQtr')
    Month= ColName('RollingMth')
    Week = ColName('RollingWk')
