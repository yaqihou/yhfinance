
class ColName:

    suffixes: tuple[str, str] = ('_cur', '_sft')

    def __init__(self, name: str, callback=None):
        self.name: str = name
        self.callback = callback

    def __call__(self, *args, **kwargs) -> str:
        if self.callback is None:
            return self.name
        else:
            return self.callback(self, *args, **kwargs)

    def __eq__(self, value: object, /) -> bool:
        if not isinstance(value, (str, ColName)):
            raise ValueError("Only support compare ColName with ColName or str")
        if isinstance(value, str):
            _cmp_str = value
        else:
            _cmp_str = value.name
        return self.name == _cmp_str
        
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

