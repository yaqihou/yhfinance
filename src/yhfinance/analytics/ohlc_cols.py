
from ._cols.base import ColName, ColBase
from ._cols.processor import ColIntra, ColToDay, ColInter, ColRolling
from ._cols.indicators import ColInd, _T_RSI

class Col(ColBase):

    Intra = ColIntra
    Inter = ColInter
    Rolling = ColRolling
    ToDay = ColToDay
    Ind = ColInd

    # TODO - Need Intraday data
    # MorningMovement   = ColName('MorningMovement')
    # AfternoonMovement = ColName('AfternoonMovement')

