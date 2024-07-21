
class ColName:

    suffixes: tuple[str, str] = ('_cur', '_sft')

    def __init__(self, name: str):
        self.name: str = name

    @property
    def sft(self) -> str:
        return self.name + self.suffixes[1]

    @property
    def cur(self) -> str:
        return self.name + self.suffixes[0]

    @property
    def gl(self) -> str:
        return self.name + 'Gl'
    
    @property
    def rtn(self) -> str:
        return self.name + "Rtn"

    @property
    def ln_rtn(self) -> str:
        return self.name + "LnRtn"

    @property
    def low(self) -> str:
        return self.name + 'Low'

    @property
    def high(self) -> str:
        return self.name + 'High'

    def __repr__(self) -> str:
        return self.name


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


class Col:

    Date     = ColName('Date')
    Datetime = ColName('Datetime')

    Open     = ColName('Open')
    High     = ColName('High')
    Low      = ColName('Low')
    Close    = ColName('Close')
    Vol      = ColName('Volume')

    OHLC: tuple[str, ...] = (Open.name, High.name, Low.name, Close.name)

    All: tuple[str, ...] = (*OHLC, Vol.name)

    Intra = ColIntra
    Inter = ColInter
    Rolling = ColRolling
    ToDay = ColToDay

    # TODO - Need Intraday data
    # MorningMovement   = ColName('MorningMovement')
    # AfternoonMovement = ColName('AfternoonMovement')

