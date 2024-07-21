
import datetime as dt

import numpy as np

def get_next_year_start(date: dt.datetime | dt.date):
    return dt.datetime(date.year+1, 1, 1)


def get_next_qtr_start(date: dt.datetime | dt.date):

    _year = date.year + 1 if date.month > 9 else date.year
    _month = (4 + 3 * ((date.month - 1) // 3)) % 12

    return dt.datetime(_year, _month, 1)


def get_next_mth_start(date: dt.datetime | dt.date):

    _year = date.year + 1 if date.month == 12 else date.year
    _month = max(1, (date.month + 1) % 12)

    return dt.datetime(_year, _month, 1)


def get_next_week_start(date: dt.datetime | dt.date):

    return date + dt.timedelta(days=7-date.weekday())

# Classic LC problem: stock 1
def get_simple_gl_rtn(arr):
    """Get the return if buy at the first day and sell at the last day"""

    gl = arr[-1] - arr[0]
    rtn = gl / arr[0] * 100
    log_rtn = np.log(arr[-1] / arr[0])

    return _GlRtn(gl=gl, rtn=rtn, log_rtn=log_rtn)
    
def get_max_simple_return(arr):
    """Get the max possible return if buy at the first day but sell at the highest day"""

    tmp = arr[1:] - arr[0]
    sell_idx = np.argmax(tmp)
    
    ret = tmp[sell_idx]
    if pct:
        ret = ret / arr[0] * 100

    return ret, sell_idx + 1  # +1 as we skip the arr[0] element

def get_max_return(arr, pct=True):
    """Return the max return Assume we only make on transaction
    """

    _min, _min_idx = arr[0], 0
    _buy_idx = _min_idx
    _sell_idx = -1
    ret = float('-inf')
    
    for idx, val in enumerate(arr[1:], 1):

        tmp = val - _min
        if tmp > ret:
            _buy_idx = _min_idx
            _sell_idx = idx
            ret = tmp
            
        if val < _min:
            _min_idx = idx
            _min = val

    if pct:
        ret = ret / _min * 100

    return ret, _buy_idx, _sell_idx


# Classic LC problem: stock 1
def get_max_loss(arr, pct=True):

    _max, _max_idx = arr[0], 0
    _buy_idx = _max_idx
    _sell_idx = -1
    
    ret = float('inf')
    
    for idx, val in enumerate(arr[1:], 1):

        tmp = val - _max
        if tmp < ret:
            _buy_idx = _max_idx
            _sell_idx = idx
            ret = tmp
            
        if val > _max:
            _max_idx = idx
            _max = val

    if pct:
        ret = ret / _max * 100
    
    return ret, _buy_idx, _sell_idx
