
from typing import Optional
from .const import Col, ColIntra, ColName, _T_RSI

__all__ = [
    '_BandMixin',
    '_RollingMixin',
    '_PricePickMixin',
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

class _PricePickMixin:

    def __init__(self,
                 *args,
                 price_col : ColName = Col.Close,
                 **kwargs
                 ):

        self.price_col = price_col
        super().__init__(*args, **kwargs)
