
import warnings
from typing import Optional

from ..ohlc_cols import Col, ColName

__all__ = [
    '_BandMixin',
    '_PeriodMixin',
    '_PriceColMixin',
    '_HistogramColorMixin'
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

class _PeriodMixin:

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
            print(type(price_col), id(price_col.__class__))
            print(ColName, id(ColName))
            print(type(price_col) is ColName)
            warnings.warn(f'price_col should be a str or ColName instance: {price_col.__class__.__name__}, {isinstance(price_col, ColName)}')
            _price_col = price_col

        # assert price_col in Col.All_PRICE
        
        self.price_col = _price_col
        super().__init__(*args, **kwargs)


class _HistogramColorMixin:
    
    @staticmethod
    def _get_histogram_color(histogram):

        histogram_color = ['#000000']
        for idx, val in enumerate(histogram[1:], 1):
            if val >= 0 and histogram[idx-1] < val:
                histogram_color.append('#26A69A')
                #print(i,'green')
            elif val >= 0 and histogram[idx-1] > val:
                histogram_color.append('#B2DFDB')
                #print(i,'faint green')
            elif val < 0 and histogram[idx-1] > val:
                #print(i,'red')
                histogram_color.append('#FF5252')
            elif val < 0 and histogram[idx-1] < val:
                #print(i,'faint red')
                histogram_color.append('#FFCDD2')
            else:
                histogram_color.append('#000000')

        return histogram_color
        
