
from collections import namedtuple
from .base import ColName

_T_MACD = namedtuple('MACD', ['EMA12', 'EMA26', 'MACD', 'Signal'])
_MACD = _T_MACD(
    ColName('MACD-EMA12'),
    ColName('MACD-EMA26'),
    ColName('MACD'),
    ColName('MACDSignal'))

def indicator_callback(self, *args, **kwargs):
    return (self.name
            + f"({','.join(map(str, args))})"
            + ' '.join([f"{k}_{v}" for k, v in kwargs])
            )
_T_RSI = namedtuple('RSI', ['AvgGain', 'AvgLoss', 'RS', 'RSI'])
_RSIWilder = _T_RSI(
    ColName('WilderRSIAvgGain', callback=indicator_callback),
    ColName('WilderRSIAvgLoss', callback=indicator_callback),
    ColName('WilderRS', callback=indicator_callback),
    ColName('WilderRSI', callback=indicator_callback))
_RSIEma = _T_RSI(
    ColName('EmaRSIAvgGain', callback=indicator_callback),
    ColName('EmaRSIAvgLoss', callback=indicator_callback),
    ColName('EmaRS', callback=indicator_callback),
    ColName('EmaRSI', callback=indicator_callback))
_RSICutler = _T_RSI(
    ColName('CutlerRSIAvgGain', callback=indicator_callback),
    ColName('CutlerRSIAvgLoss', callback=indicator_callback),
    ColName('CutlerRS', callback=indicator_callback),
    ColName('CutlerRSI', callback=indicator_callback))

_T_SUPERTREND = namedtuple('SuperTrend', ['Up', 'Dn', 'Final', 'Mode'])
_SUPERTREND = _T_SUPERTREND(
    ColName('SupertrendUp', callback=indicator_callback),
    ColName('SupertrendDn', callback=indicator_callback),
    ColName('Supertrend', callback=indicator_callback),
    ColName('SupertrendMode')
)

_T_AROON = namedtuple('Aroon', ['Up', 'Dn', 'Oscillator'])
_AROON = _T_AROON(
    ColName('AroonUp', callback=indicator_callback),
    ColName('AroonDn', callback=indicator_callback),
    ColName('AroonOscillator', callback=indicator_callback),
)

_T_STARC = namedtuple('STARCBand', ['SMA', 'ATR', 'Up', 'Dn', 'STARC'])
_STARC = _T_STARC(
    ColName('STARC-SMA', callback=indicator_callback),
    ColName('STARC-ATR', callback=indicator_callback),
    ColName('STARCUp', callback=indicator_callback),
    ColName('STARCDn', callback=indicator_callback),
    ColName('STARC', callback=indicator_callback),
)

_T_AWESOME_OSCILLATOR = namedtuple('AwesomeOscillator', ['Fast', 'Slow', 'AO'])
_AWESOME_OSCILLATOR = _T_AWESOME_OSCILLATOR(
    ColName('AO-SMAFast', callback=indicator_callback),
    ColName('AO-SMASlow', callback=indicator_callback),
    ColName('AO', callback=indicator_callback),
)

_T_BOLLINGER_BAND = namedtuple('BollingerBand', ['SMA', 'Std', 'Up', 'Dn', 'BB'])
_BOLLINGER_BAND = _T_BOLLINGER_BAND(
    ColName('BB-SMA', callback=indicator_callback),
    ColName('BB-Std', callback=indicator_callback),
    ColName('BBUp', callback=indicator_callback),
    ColName('BBDn', callback=indicator_callback),
    ColName('BollingerBand', callback=indicator_callback),
)
    
# TODO - could further divided into MOmentum / etc.
class ColInd:

    # TODO - add function to list by categories

    SMA = ColName('SMA', callback=indicator_callback)
    EMA = ColName('EMA', callback=indicator_callback)
    SMMA = ColName('SMMA', callback=indicator_callback)

    AwesomeOscillator = _AWESOME_OSCILLATOR
    BollingerBand = _BOLLINGER_BAND

    MACD = _MACD
    RSIWilder = _RSIWilder
    RSIEma = _RSIEma
    RSICutler = _RSICutler

    Aroon = _AROON

    TrueRange = ColName('TR')
    AvgTrueRange = ColName('ATR', callback=indicator_callback)
    SuperTrend = _SUPERTREND
    STARC = _STARC
    
