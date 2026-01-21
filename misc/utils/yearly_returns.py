import pandas as pd


def calculate_yearly_returns(data: pd.DataFrame) -> pd.DataFrame:
    temp = data["Close"].resample("YE").ffill()
    return temp.pct_change().dropna()
