
from ._cols.base import ColName
from ._cols.processor import ColIntra, ColToDay, ColInter, ColRolling
from ._cols.indicators import ColInd, _T_RSI

class Col:

    Date     = ColName('Date')
    Datetime = ColName('Datetime')

    Open     = ColName('Open')
    High     = ColName('High')
    Low      = ColName('Low')
    Close    = ColName('Close')
    Median   = ColName('MedianHL')
    Typical  = ColName('TypicalHLC')
    Avg      = ColName('AvgOHLC')

    Vol      = ColName('Volume')

    OHLC: tuple[str, ...] = (Open.name, High.name, Low.name, Close.name)

    All_PRICE: tuple[str, ...] = (*OHLC, Median.name, Typical.name, Avg.name)
    All_PRICE_WITH_VOL: tuple[str, ...] = (*OHLC, Median.name, Typical.name, Avg.name, Vol.name)

    Intra = ColIntra
    Inter = ColInter
    Rolling = ColRolling
    ToDay = ColToDay
    Ind = ColInd

    # TODO - Need Intraday data
    # MorningMovement   = ColName('MorningMovement')
    # AfternoonMovement = ColName('AfternoonMovement')

