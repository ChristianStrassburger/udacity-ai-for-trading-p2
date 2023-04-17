import numpy as np
import pandas as pd
from datetime import datetime

from pandas.core.window.rolling import Window


#Pandas.DataFrame.rolling

# print()
# print(datetime.strptime('10/10/2018', '%m/%d/%Y'))

# dates = pd.date_range(datetime.strptime('10/10/2018', '%m/%d/%Y'), periods=11, freq='D')
# close_prices = np.arange(len(dates))
# close = pd.Series(close_prices, dates)

dates = pd.date_range('10/10/2018', periods=11, freq='D')
df = np.arange(len(dates))

df_aapl = pd.DataFrame(
    {
        "date": dates,
        "ticker": "AAPL",
        "adj_close": df
    }
)

df_tsla = pd.DataFrame(
    {
        "date": dates,
        "ticker": "TSLA",
        "adj_close": df + 2
    }
)

df_abcd = pd.DataFrame(
    {
        "date": dates,
        "ticker": "ABCD",
        "adj_close": df + 4
    }
)

df_nvdi = pd.DataFrame(
    {
        "date": dates,
        "ticker": "AVDI",
        "adj_close": df + 6
    }
)

df_zztl = pd.DataFrame(
    {
        "date": dates,
        "ticker": "ZZTL",
        "adj_close": df + 8
    }
)

df_full = df_aapl.append(df_tsla, ignore_index=True)
df_full = df_full.append(df_abcd, ignore_index=True)
df_full = df_full.append(df_zztl, ignore_index=True)
df_full = df_full.append(df_nvdi, ignore_index=True)
df_full = df_full.set_index("date", drop=True)
close = df_full.reset_index().pivot(index='date', columns='ticker', values='adj_close')
print()
print("close")
print(close)



# print()
# print("close")
# print(close)

#rolling = close.rolling(window = 3)
# print()
# print("close.rolling")
# print(rolling)

# sum = close.rolling(window = 3).sum()
# print()
# print("sum")
# print(sum)

# min = close.rolling(window = 3).min()
# print()
# print("min")
# print(min)


#Quiz: Calculate Simple Moving Average
def calculate_simple_moving_average(rolling_window, close):
    """
    Compute the simple moving average.
    
    Parameters
    ----------
    rolling_window: int
        Rolling window length
    close : DataFrame
        Close prices for each ticker and date
    
    Returns
    -------
    simple_moving_average : DataFrame
        Simple moving average for each ticker and date
    """
    # TODO: Implement Function
    
    print()
    print(close)

    simple_moving_average = close.rolling(window=rolling_window).mean()
    print()
    print(simple_moving_average)

    return simple_moving_average

print()
calculate_simple_moving_average = calculate_simple_moving_average(rolling_window=3, close=close)
print("calculate_simple_moving_average")



### Tests
# print()
# print("ewm")
# ewm = close.ewm(alpha=0.9, adjust=True).mean()

# print(ewm)


