
from .base import ColName

class ColIntra:
    # Intra-tick features
    Return           = ColName('IntraOpenClose')
    Swing            = ColName('IntraSwing')
    SwingGlRatio     = ColName('GlSwingRatio')

    GainStreak = ColName('IntraGainStreak')
    LossStreak = ColName('IntraLossStreak')
    Streak = ColName('IntraStreak')


class ColToDay:

    Year   = ColName('YTD')
    Qtr   = ColName('QTD')
    Month = ColName('MTD')
    Week = ColName('WTD')

class ColInter:
    # Inter-tick features
    CloseOpenSpread = ColName('InterSpread')
    CloseReturn     = ColName('InterClose')
    OpenReturn      = ColName('InterOpen')
    OpenCloseReturn = ColName('InterOpenClose')
    
    CloseGainStreak = ColName('InterCloseGainStreak')
    CloseLossStreak = ColName('InterCloseLossStreak')
    CloseStreak = ColName('InterCloseStreak')

    OpenGainStreak = ColName('InterOpenGainStreak')
    OpenLossStreak = ColName('InterOpenLossStreak')
    OpenStreak = ColName('InterOpenStreak')

class ColRolling:
    # Rolling Window
    Year = ColName('Rolling52Wk')
    Qtr= ColName('RollingQtr')
    Month= ColName('RollingMth')
    Week = ColName('RollingWk')
