
import pandas as pd
import datetime as dt

def parse_input_datetime(
        _input: str | dt.datetime | dt.date | int,
        return_pydatetime: bool = True
) -> pd.Timestamp | dt.datetime:

    if isinstance(_input, int):
        ret = pd.to_datetime(_input, unit='s')
    else:
        ret = pd.to_datetime(_input)

    return ret.to_pydatetime() if return_pydatetime else ret
