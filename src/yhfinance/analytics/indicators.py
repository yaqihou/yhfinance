
from typing import Literal

from ._indicators._base import _BaseIndicator
from ._indicators.basics import IndEMA, IndSMA, IndSMMA
from ._indicators.momentum_oscillator import IndAwesomeOscillator, IndMACD, IndWilderRSI, IndEmaRSI, IndCutlerRSI
from ._indicators.moving_average_band import IndAvgTrueRange, IndATRBand, IndBollingerBand, IndBollingerBandModified, IndStarcBand
from ._indicators.trend import IndAroon, IndSupertrend
from ._indicators.volatility import IndTrueRange
from ._indicators.volume import IndMoneyFlowIndex


__all__ = [
    'IndSMA',
    'IndMACD',
    'IndWilderRSI', 'IndEmaRSI', 'IndCutlerRSI',
    'IndTrueRange', 'IndAvgTrueRange', 'IndATRBand', 'IndStarcBand',
    'IndSupertrend',
    'IndAroon',
    'IndAwesomeOscillator',
    'IndBollingerBand',
    'IndBollingerBandModified',
    'IndMoneyFlowIndex'
]

# TODO - add plotter configs into each class so that could be used outside
# TODO - add related indicators as class property

class IndicatorProperty:

    category: Literal['Band', 'Oscillator', 'Price', 'Undefined']
    abbreviation: str
    fullname: str
    created_year: int   # may need to exclude this feature if it is too old, just nice to have it
    fundamental: list[str]



# -----------------------------------
# Implementation starts below
