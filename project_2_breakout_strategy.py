import pandas as pd
import numpy as np
from pandas.tseries import offsets
# import helper
# import project_helper
# import project_tests

#Market Data
# df_original = pd.read_csv('../../data/project_2/eod-quotemedia.csv', parse_dates=['date'], index_col=False)

# # Add TB sector to the market
# df = df_original
# df = pd.concat([df] + project_helper.generate_tb_sector(df[df['ticker'] == 'AAPL']['date']), ignore_index=True)

# close = df.reset_index().pivot(index='date', columns='ticker', values='adj_close')
# high = df.reset_index().pivot(index='date', columns='ticker', values='adj_high')
# low = df.reset_index().pivot(index='date', columns='ticker', values='adj_low')

#print('Loaded Data')

#Stock Example
apple_ticker = 'AAPL'
#project_helper.plot_stock(close[apple_ticker], '{} Stock'.format(apple_ticker))

#Custom data 
dates = pd.date_range('09/11/2004', periods=4, freq='D')
df = np.arange(len(dates))

df_aapl = pd.DataFrame(
    {
        "date": dates,
        "ticker": "AAPL",
        "adj_low": [15.67180000, 27.18340000, 28.25030000, 86.37250000]
    }
)

df_tsla = pd.DataFrame(
    {
        "date": dates,
        "ticker": "TSLA",
        "adj_low": [75.13920000, 12.34530000, 24.28540000, 32.22300000]
    }
)

df_abcd = pd.DataFrame(
    {
        "date": dates,
        "ticker": "ABCD",
        "adj_low": [34.05270000, 95.93730000, 23.29320000, 38.41070000]
    }
)


df_full = df_aapl.append(df_tsla, ignore_index=True)
df_full = df_full.append(df_abcd, ignore_index=True)

df_full = df_full.set_index("date", drop=True)
low = df_full.reset_index().pivot(index='date', columns='ticker', values='adj_low')
high = low + 25
close = low + 10


# print()
# print("low")
# print(low)

# print()
# print("high")
# print(high)
#end custom data




# #Compute the Highs and Lows in a Window
# high
#                    GSD        QIDI        WSGN
# 2004-09-11 35.44110000 34.17990000 34.02230000
# 2004-09-12 92.11310000 91.05430000 90.95720000
# 2004-09-13 57.97080000 57.78140000 58.19820000
# 2004-09-14 34.17050000 92.45300000 58.51070000

# low
#                    GSD        QIDI        WSGN
# 2004-09-11 15.67180000 75.13920000 34.05270000
# 2004-09-12 27.18340000 12.34530000 95.93730000
# 2004-09-13 28.25030000 24.28540000 23.29320000
# 2004-09-14 86.37250000 32.22300000 38.41070000

# lookback_days
# 2
def get_high_lows_lookback(high, low, lookback_days): #Ok works!!!
    """
    Get the highs and lows in a lookback window.
    
    Parameters
    ----------
    high : DataFrame
        High price for each ticker and date
    low : DataFrame
        Low price for each ticker and date
    lookback_days : int
        The number of days to look back
    
    Returns
    -------
    lookback_high : DataFrame
        Lookback high price for each ticker and date
    lookback_low : DataFrame
        Lookback low price for each ticker and date
    """
    #TODO: Implement function
    # print()
    # print("low")
    # print(low)

    # print()
    # print("high")
    # print(high)

    lookback_high = high.shift(1).rolling(window=lookback_days).max()
    lookback_low = low.shift(1).rolling(window=lookback_days).min()

    # print()
    # print(f"type: {type(lookback_high)}")
    # print(lookback_high)

    # print()
    # print(f"type: {type(lookback_low)}")
    # print(lookback_low)

    return lookback_high, lookback_low

#project_tests.test_get_high_lows_lookback(get_high_lows_lookback)
#get_high_lows_lookback(high=high, low=low, lookback_days=2) #Ok - works!!


# #View Data
# lookback_days = 50
lookback_days = 2
lookback_high, lookback_low = get_high_lows_lookback(high, low, lookback_days) #Ok - works!!
# project_helper.plot_high_low(
#     close[apple_ticker],
#     lookback_high[apple_ticker],
#     lookback_low[apple_ticker],
#     'High and Low of {} Stock'.format(apple_ticker))

#Compute Long and Short Signals
def get_long_short(close, lookback_high, lookback_low):
    """
    Generate the signals long, short, and do nothing.
    
    Parameters
    ----------
    close : DataFrame
        Close price for each ticker and date
    lookback_high : DataFrame
        Lookback high price for each ticker and date
    lookback_low : DataFrame
        Lookback low price for each ticker and date
    
    Returns
    -------
    long_short : DataFrame
        The long, short, and do nothing signals for each ticker and date
    """
    #TODO: Implement function
    # print("close")
    # print(close.head(5))
    
    # print()
    # print("lookback_high")
    # print(lookback_high.head(5))
    
    # print()
    # print("lookback_low")
    # print(lookback_low.head(5))
    long_short = close.copy()
    long_short = long_short.fillna(0)
    long_short = long_short.astype(np.int64, copy=False)

    # print()
    # print("long_short")
    # print(long_short)
    
    #-1 	Low > Close Price
    # 1 	High < Close Price
    # 0 	Otherwise
    low_signal = lookback_low > close
    high_signal = lookback_high < close
    default_signal = ~(low_signal | high_signal)

    # print()
    # print("default_signal")
    # print(default_signal)

    long_short[low_signal] = -1
    long_short[high_signal] = 1
    long_short[default_signal] = 0

    # print()
    # print("long_short")
    # print(long_short)

    return long_short

# project_tests.test_get_long_short(get_long_short)

# #View Data
signal = get_long_short(close, lookback_high, lookback_low)
# project_helper.plot_signal(
#     close[apple_ticker],
#     signal[apple_ticker],
#     'Long and Short of {} Stock'.format(apple_ticker))

# #Filter Signals
#               RFJE  VMP  MVL
# 2006-03-07     0    0    0
# 2006-03-08    -1   -1   -1
# 2006-03-09     1    0   -1
# 2006-03-10     0    0    0
# 2006-03-11     1    0    0
# 2006-03-12     0    1    0
# 2006-03-13     0    0    1
# 2006-03-14     0   -1    1
# 2006-03-15    -1    0    0
# 2006-03-16     0    0    0
#chs data
signal_dates = pd.date_range('03/07/2006', periods=10, freq='D')
df = np.arange(len(dates))

df_rjfe = pd.DataFrame(
    {
        "date": signal_dates,
        "ticker": "RFJE",
        "signal": [0, -1, 1, 0, 1, 0 , 0, 0, -1, 0]
    }
)
df_mvl = pd.DataFrame(
    {
        "date": signal_dates,
        "ticker": "MVL",
        "signal": [0, -1, -1, 0, 0, 0, 1, 1, 0, 0]
    }
)

df_vmp = pd.DataFrame(
    {
        "date": signal_dates,
        "ticker": "VMP",
        "signal": [0, -1, 0, 0, 0, 1, 0, -1, 0, 0]
    }
)

df_signal_full = df_rjfe.append(df_vmp, ignore_index=True)
df_signal_full = df_signal_full.append(df_mvl, ignore_index=True)
df_signal_full = df_signal_full.set_index("date", drop=True)

test_signals = df_signal_full.reset_index().pivot(index='date', columns='ticker', values='signal')
test_signals = test_signals[["RFJE", "VMP", "MVL"]]
#end data


def clear_signals(signals, window_size):
    """
    Clear out signals in a Series of just long or short signals.
    
    Remove the number of signals down to 1 within the window size time period.
    
    Parameters
    ----------
    signals : Pandas Series
        The long, short, or do nothing signals
    window_size : int
        The number of days to have a single signal       
    
    Returns
    -------
    signals : Pandas Series
        Signals with the signals removed from the window size
    """
    # Start with buffer of window size
    # This handles the edge case of calculating past_signal in the beginning
    clean_signals = [0]*window_size
    
    for signal_i, current_signal in enumerate(signals):
        # Check if there was a signal in the past window_size of days
        has_past_signal = bool(sum(clean_signals[signal_i:signal_i+window_size]))
        # Use the current signal if there's no past signal, else 0/False
        clean_signals.append(not has_past_signal and current_signal)
        
    # Remove buffer
    clean_signals = clean_signals[window_size:]

    # Return the signals as a Series of Ints
    return pd.Series(np.array(clean_signals).astype(np.int), signals.index)


def filter_signals(signal, lookahead_days): #ok, works!!!
    """
    Filter out signals in a DataFrame.
    
    Parameters
    ----------
    signal : DataFrame
        The long, short, and do nothing signals for each ticker and date
    lookahead_days : int
        The number of days to look ahead
    
    Returns
    -------
    filtered_signal : DataFrame
        The filtered long, short, and do nothing signals for each ticker and date
    """
    #TODO: Implement function
    filtered_signal = signal.copy()

    # print()
    # print("lookahead_days")
    # print(lookahead_days)

    # print()
    # print("signal")
    # print(filtered_signal)

    low_signal = filtered_signal[filtered_signal == -1]
    low_signal = low_signal.fillna(0)

    # print()
    # print("low_signal")
    # print(low_signal)

    high_signal = filtered_signal[filtered_signal == 1]
    high_signal = high_signal.fillna(0)

    # print()
    # print("high_signal")
    # print(high_signal)

    for column_name in low_signal:
        # print()
        # print(F"column: " + column_name)
        # print(f"column type: {type(low_signal[column_name])}")
        signal_col = low_signal[column_name]
        cleared_signal_col = clear_signals(signals=signal_col, window_size=lookahead_days)
        #print("cleared_signal_col")
        #print(cleared_signal_col)
        low_signal[column_name] = cleared_signal_col
        #top_stocks.loc[nlargest.name, nlargest.index] = 1

    for column_name in high_signal:
        # print()
        # print(F"column: " + column_name)
        # print(f"column type: {type(high_signal[column_name])}")
        signal_col = high_signal[column_name]
        cleared_signal_col = clear_signals(signals=signal_col, window_size=lookahead_days)
        #print("cleared_signal_col")
        #print(cleared_signal_col)
        high_signal[column_name] = cleared_signal_col

    filtered_signal = high_signal + low_signal

    # print()
    # print("signal")
    # print(filtered_signal)

    return filtered_signal

# project_tests.test_filter_signals(filter_signals)

#[1, 0, 1, 0, 1, 0, -1, -1]
#[1, 0, 0, 0, 1, 0, -1, -1]
#[1, 0, 0, 0, 1, 0, -1, -1]
#[1, 0, 0, 0, 1, 0, -1, -1]
#[1, 0, 0, 0, 1, 0, -1, -1]
#[1, 0, 0, 0, 1, 0, -1, -1]
#[1, 0, 0, 0, 1, 0, -1, 0]

#[1, 0, 0, 0, 1, 0, -1, 0]
#result
#[1, 0, 0, 0, 1, 0, -1, 0]


# #View Data
signal_result = filter_signals(test_signals, 3) #ok, works!!!
#signal_5 = filter_signals(signal, 5)
# signal_10 = filter_signals(signal, 10)
# signal_20 = filter_signals(signal, 20)
# for signal_data, signal_days in [(signal_5, 5), (signal_10, 10), (signal_20, 20)]:
#     project_helper.plot_signal(
#         close[apple_ticker],
#         signal_data[apple_ticker],
#         'Long and Short of {} Stock with {} day signal window'.format(apple_ticker, signal_days))




# #Lookahead Close Prices
def get_lookahead_prices(close, lookahead_days): #ok, works!!!
    """
    Get the lookahead prices for `lookahead_days` number of days.
    
    Parameters
    ----------
    close : DataFrame
        Close price for each ticker and date
    lookahead_days : int
        The number of days to look ahead
    
    Returns
    -------
    lookahead_prices : DataFrame
        The lookahead prices for each ticker and date
    """
    #TODO: Implement function
    # print("close")
    # print(close)
    
    # print()
    # print("lookahead_days")
    # print(lookahead_days)

    lookahead_prices = close.shift(-1 * lookahead_days)

    # print()
    # print("lookahead_prices")
    # print(lookahead_prices)

    return lookahead_prices

# project_tests.test_get_lookahead_prices(get_lookahead_prices)

# #View Data
lookahead_2 = get_lookahead_prices(close, 2)
#lookahead_5 = get_lookahead_prices(close, 5)
# lookahead_10 = get_lookahead_prices(close, 10)
# lookahead_20 = get_lookahead_prices(close, 20)
# project_helper.plot_lookahead_prices(
#     close[apple_ticker].iloc[150:250],
#     [
#         (lookahead_5[apple_ticker].iloc[150:250], 5),
#         (lookahead_10[apple_ticker].iloc[150:250], 10),
#         (lookahead_20[apple_ticker].iloc[150:250], 20)],
#     '5, 10, and 20 day Lookahead Prices for Slice of {} Stock'.format(apple_ticker))


# #Lookahead Price Returns
def get_return_lookahead(close, lookahead_prices):
    """
    Calculate the log returns from the lookahead days to the signal day.
    
    Parameters
    ----------
    close : DataFrame
        Close price for each ticker and date
    lookahead_prices : DataFrame
        The lookahead prices for each ticker and date
    
    Returns
    -------
    lookahead_returns : DataFrame
        The lookahead log returns for each ticker and date
    """
    #TODO: Implement function
    # print("close")
    # print(close)
    
    # print()
    # print("lookahead_prices")
    # print(lookahead_prices)

    raw_return = (lookahead_prices - close) / close
    # print()
    # print("raw_return")
    # print(raw_return)

    lookahead_returns = np.log(raw_return + 1)
    # print()
    # print("lookahead log_returns")
    # print(lookahead_returns)    

    return lookahead_returns


lookahead_returns = get_return_lookahead(close=close, lookahead_prices=lookahead_2)
# project_tests.test_get_return_lookahead(get_return_lookahead)

# #View Data
# price_return_5 = get_return_lookahead(close, lookahead_5)
# price_return_10 = get_return_lookahead(close, lookahead_10)
# price_return_20 = get_return_lookahead(close, lookahead_20)
# project_helper.plot_price_returns(
#     close[apple_ticker].iloc[150:250],
#     [
#         (price_return_5[apple_ticker].iloc[150:250], 5),
#         (price_return_10[apple_ticker].iloc[150:250], 10),
#         (price_return_20[apple_ticker].iloc[150:250], 20)],
#     '5, 10, and 20 day Lookahead Returns for Slice {} Stock'.format(apple_ticker))


# #Compute the Signal Return
# INPUT signal:
#             EEGD  YZD  VOOB
# 2011-10-09     0    0     0
# 2011-10-10    -1   -1    -1
# 2011-10-11     1    0     0
# 2011-10-12     0    0     0
# 2011-10-13     0    1     0

# INPUT lookahead_returns:
#                  EEGD         YZD        VOOB
# 2011-10-09 0.88702896  0.96521098  0.65854789
# 2011-10-10 1.13391240  0.87420969 -0.53914925
# 2011-10-11 0.35450805 -0.56900529 -0.64808965
# 2011-10-12 0.38572896 -0.94655617  0.12356438
# 2011-10-13        nan         nan         nan

# OUTPUT signal_return:
#                  EEGD         YZD        VOOB
# 2011-10-09 0.88702896  0.96521098  0.65854789
# 2011-10-10 1.13391240  0.87420969 -0.53914925
# 2011-10-11 0.35450805 -0.56900529 -0.64808965
# 2011-10-12 0.38572896 -0.94655617  0.12356438
# 2011-10-13        nan         nan         nan

# EXPECTED OUTPUT FOR signal_return:
#                   EEGD         YZD       VOOB
# 2011-10-09  0.00000000  0.00000000 0.00000000
# 2011-10-10 -1.13391240 -0.87420969 0.53914925
# 2011-10-11  0.35450805  0.00000000 0.00000000
# 2011-10-12  0.00000000  0.00000000 0.00000000
# 2011-10-13         nan         nan        nan
def get_signal_return(signal, lookahead_returns):
    """
    Compute the signal returns.
    
    Parameters
    ----------
    signal : DataFrame
        The long, short, and do nothing signals for each ticker and date
    lookahead_returns : DataFrame
        The lookahead log returns for each ticker and date
    
    Returns
    -------
    signal_return : DataFrame
        Signal returns for each ticker and date
    """
    #TODO: Implement function
    # print("signal")
    # print(signal)
    
    # print()
    # print("lookahead_returns")
    #print(lookahead_returns)

    signal_return = signal * lookahead_returns
    
    return signal_return

# project_tests.test_get_signal_return(get_signal_return)
get_signal_return(signal=signal_result, lookahead_returns=lookahead_returns)

# #View Data
# title_string = '{} day LookaheadSignal Returns for {} Stock'
# signal_return_5 = get_signal_return(signal_5, price_return_5)
# signal_return_10 = get_signal_return(signal_10, price_return_10)
# signal_return_20 = get_signal_return(signal_20, price_return_20)
# project_helper.plot_signal_returns(
#     close[apple_ticker],
#     [
#         (signal_return_5[apple_ticker], signal_5[apple_ticker], 5),
#         (signal_return_10[apple_ticker], signal_10[apple_ticker], 10),
#         (signal_return_20[apple_ticker], signal_20[apple_ticker], 20)],
#     [title_string.format(5, apple_ticker), title_string.format(10, apple_ticker), title_string.format(20, apple_ticker)])


# #Test for Significance
# #Histogram
# project_helper.plot_signal_histograms(
#     [signal_return_5, signal_return_10, signal_return_20],
#     'Signal Return',
#     ('5 Days', '10 Days', '20 Days'))


# #Outliers
# project_helper.plot_signal_to_normal_histograms(
#     [signal_return_5, signal_return_10, signal_return_20],
#     'Signal Return',
#     ('5 Days', '10 Days', '20 Days'))


# #Kolmogorov-Smirnov Test
# # Filter out returns that don't have a long or short signal.
# long_short_signal_returns_5 = signal_return_5[signal_5 != 0].stack()
# long_short_signal_returns_10 = signal_return_10[signal_10 != 0].stack()
# long_short_signal_returns_20 = signal_return_20[signal_20 != 0].stack()

# # Get just ticker and signal return
# long_short_signal_returns_5 = long_short_signal_returns_5.reset_index().iloc[:, [1,2]]
# long_short_signal_returns_5.columns = ['ticker', 'signal_return']
# long_short_signal_returns_10 = long_short_signal_returns_10.reset_index().iloc[:, [1,2]]
# long_short_signal_returns_10.columns = ['ticker', 'signal_return']
# long_short_signal_returns_20 = long_short_signal_returns_20.reset_index().iloc[:, [1,2]]
# long_short_signal_returns_20.columns = ['ticker', 'signal_return']

# # View some of the data
# long_short_signal_returns_5.head(10)

from scipy.stats import kstest


# long_short_signal_returns
#    ticker  signal_return
# 0     DTM     0.12000000
# 1     HFV    -0.83000000
# 2     HMI     0.37000000
# 3     DTM     0.83000000
# 4     HFV    -0.34000000
# 5     HMI     0.27000000
# 6     DTM    -0.68000000
# 7     HFV     0.29000000
# 8     HMI     0.69000000
# 9     DTM     0.57000000
# 10    HFV     0.39000000
# 11    HMI     0.56000000
# 12    DTM    -0.97000000
# 13    HFV    -0.72000000
# 14    HMI     0.26000000

ls_ticker = ["DTM", "HFV", "HMI", "DTM", "HFV", "HMI", "DTM", "HFV", "HMI", "DTM", "HFV", "HMI", "DTM", "HFV", "HMI"]
ls_sr = [0.12000000, -0.83000000, 0.37000000, 0.83000000, -0.34000000, 0.27000000, -0.68000000, 0.29000000, 0.69000000, 0.57000000, 0.39000000, 0.56000000, -0.97000000, -0.72000000, 0.26000000]

long_short_signal_returns = pd.DataFrame(
    {
        "ticker": ls_ticker,
        "signal_return": ls_sr
    })

def calculate_kstest(long_short_signal_returns):
    """
    Calculate the KS-Test against the signal returns with a long or short signal.
    
    Parameters
    ----------
    long_short_signal_returns : DataFrame
        The signal returns which have a signal.
        This DataFrame contains two columns, "ticker" and "signal_return"
    
    Returns
    -------
    ks_values : Pandas Series
        KS static for all the tickers
    p_values : Pandas Series
        P value for all the tickers
    """
    #TODO: Implement function
    # print("long_short_signal_returns")
    # print(long_short_signal_returns)

    # print()
    # print("signal_return")
    # print(signal_return)

    # signal_return.groupby("ticker")

    ticker_group = long_short_signal_returns.groupby("ticker")

    signal_return = long_short_signal_returns["signal_return"]
    normal_args = (signal_return.mean(), signal_return.std(ddof=0))

    # print()
    # print("ticker_group")
    # print(f"type: {type(ticker_group)}")
    # print(ticker_group.count())
    ticker_group_ksresult = ticker_group.apply(lambda x: kstest(x["signal_return"], 'norm', normal_args ))

    ks_values = ticker_group_ksresult.apply(lambda x: x[0])
    p_values = ticker_group_ksresult.apply(lambda x: x[1])

    # print()
    # print("ticker_group_ksresult")
    # print(ticker_group_ksresult)

    # print()
    # print("ks_values")
    # print(f"type: {type(ks_values)}")
    # print(ks_values)

    # print()
    # print("p_values")
    # print(f"type: {type(p_values)}")
    # print(p_values)

    # normal_args = (signal_return.mean(), signal_return.std())
    # t_stat, p_value = kstest(signal_return, 'norm', normal_args)
    # print("Test statistic: {}, p-value: {}".format(t_stat, p_value))
    # #print("Is the distribution Likely Normal? {}".format(p_value > p_level))
    # #return p_value > p_level

    return ks_values, p_values


# project_tests.test_calculate_kstest(calculate_kstest)
# OUTPUT ks_values:
# ticker
# DZK    0.25477574
# HJBK   0.23881184
# XTMY   0.22600258
# dtype: float64

# OUTPUT p_values:
# ticker
# DZK    0.90169046
# HJBK   0.93796862
# XTMY   0.96041495
# dtype: float64

# EXPECTED OUTPUT FOR ks_values:
# XTMY   0.29787827
# DZK    0.35221525
# HJBK   0.63919407
# dtype: float64

# EXPECTED OUTPUT FOR p_values:
# XTMY   0.69536353
# DZK    0.46493498
# HJBK   0.01650327
# dtype: float64
(ks_values, p_values) = calculate_kstest(long_short_signal_returns)

# #View Data
# ks_values_5, p_values_5 = calculate_kstest(long_short_signal_returns_5)
# ks_values_10, p_values_10 = calculate_kstest(long_short_signal_returns_10)
# ks_values_20, p_values_20 = calculate_kstest(long_short_signal_returns_20)

# print('ks_values_5')
# print(ks_values_5.head(10))
# print('p_values_5')
# print(p_values_5.head(10))


# #Find Outliers
# ks_values
# KQS    0.20326939
# GHC    0.34826827
# DFVF   0.60256811
# dtype: float64

# p_values
# KQS    0.98593727
# GHC    0.48009144
# DFVF   0.02898631
# dtype: float64

# ks_threshold
# 0.5

# pvalue_threshold
# 0.05
def find_outliers(ks_values, p_values, ks_threshold, pvalue_threshold=0.05):
    """
    Find outlying symbols using KS values and P-values
    
    Parameters
    ----------
    ks_values : Pandas Series
        KS static for all the tickers
    p_values : Pandas Series
        P value for all the tickers
    ks_threshold : float
        The threshold for the KS statistic
    pvalue_threshold : float
        The threshold for the p-value
    
    Returns
    -------
    outliers : set of str
        Symbols that are outliers
    """
    #TODO: Implement function
    print("ks_values")
    print(ks_values)
    
    print()
    print("p_values")
    print(p_values)
    
    print()
    print("ks_threshold")
    print(ks_threshold)
    
    print()
    print("pvalue_threshold")
    print(pvalue_threshold)

    p_check = p_values < pvalue_threshold

    print()
    print("p_check")
    print(p_check)

    ks_check = ks_values > ks_threshold
    print()
    print("ks_check")
    print(ks_check)

    p_ks_check = p_check & ks_check
    outliers = set(p_ks_check[p_ks_check].index)

    # print()
    # print(p_ks_check[p_ks_check].index)
    # print(f"type: {type(outliers)}")

    print()
    print("outliers")
    print(f"type: {type(outliers)}")
    print(outliers)

    return outliers


# project_tests.test_find_outliers(find_outliers)
find_outliers(ks_values=ks_values, p_values=p_values, ks_threshold=0.5)



# #View Data
# ks_threshold = 0.8
# outliers_5 = find_outliers(ks_values_5, p_values_5, ks_threshold)
# outliers_10 = find_outliers(ks_values_10, p_values_10, ks_threshold)
# outliers_20 = find_outliers(ks_values_20, p_values_20, ks_threshold)

# outlier_tickers = outliers_5.union(outliers_10).union(outliers_20)
# print('{} Outliers Found:\n{}'.format(len(outlier_tickers), ', '.join(list(outlier_tickers))))

# #Show Significance without Outliers
# good_tickers = list(set(close.columns) - outlier_tickers)

# project_helper.plot_signal_to_normal_histograms(
#     [signal_return_5[good_tickers], signal_return_10[good_tickers], signal_return_20[good_tickers]],
#     'Signal Return Without Outliers',
#     ('5 Days', '10 Days', '20 Days'))




