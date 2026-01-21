import pandas as pd


def calculate_monthly_returns(data: pd.DataFrame) -> pd.DataFrame:
    temp = data["Close"].resample("ME").ffill()
    return temp.pct_change().dropna()
