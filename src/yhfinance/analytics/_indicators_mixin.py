
from typing import Optional
from .ohlc_cols import Col, ColIntra, ColName, _T_RSI

__all__ = [
    '_BandMixin',
    '_RollingMixin',
    '_PriceColMixin',
]

class _BandMixin:

    def __init__(self,
                 *args,
                 multiplier    : int           = 3,
                 multiplier_dn : Optional[int] = None,  # if None, the same as multiplier
                 **kwargs
                 ):

        self.multiplier = multiplier
        self._multi_up = multiplier
        self._multi_dn = multiplier_dn or self._multi_up

        super().__init__(*args, **kwargs)

class _RollingMixin:

    def __init__(self,
                 *args,
                 period : int = 7,  # if None, the same as multiplier
                 **kwargs
                 ):

        self.period = period
        super().__init__(*args, **kwargs)

class _PriceColMixin:

    def __init__(self,
                 *args,
                 price_col : ColName | str = Col.Close,
                 **kwargs
                 ):

        if isinstance(price_col, str):
            _price_col = ColName(price_col)
        elif isinstance(price_col, ColName):
            _price_col = price_col
        else:
            raise ValueError(f'price_col should be a str or ColName instance: {price_col.__class__.__name__}')

        assert price_col in Col.All_PRICE
        
        self.price_col = _price_col
        super().__init__(*args, **kwargs)
