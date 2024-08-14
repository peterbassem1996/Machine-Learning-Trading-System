import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

from util import get_data, plot_data
import datetime as dt

def author():
    return "pdawoud37"

def study_group():
    return "pdawoud37"

def bollinger_bands(
        prices,
        window=20,
        num_std_dev=2,
        symbol="JPM",
):


    # Calculate the moving average (middle band)
    prices['SMA'] = prices[symbol].rolling(window=window).mean()

    # Calculate the standard deviation
    rolling_std = prices[symbol].rolling(window=window).std()

    # Calculate the upper and lower bands
    prices['Upper Band'] = prices['SMA'] + (rolling_std * num_std_dev)
    prices['Lower Band'] = prices['SMA'] - (rolling_std * num_std_dev)

    return prices


def macd(prices, short_span=12, long_span=26, signal_span=9, symbol='JPM'):
    # Calculate the short-term and long-term EMAs
    prices['EMA_short'] = prices[symbol].ewm(span=short_span, adjust=False, min_periods=short_span).mean()
    prices['EMA_long'] = prices[symbol].ewm(span=long_span, adjust=False, min_periods=long_span).mean()

    # Calculate the MACD line
    prices['MACD'] = prices['EMA_short'] - prices['EMA_long']

    # Calculate the Signal line
    prices['Signal_Line'] = prices['MACD'].ewm(span=signal_span, adjust=False).mean()

    return prices


def rsi(prices, window=14, symbol='JPM'):
    # Calculate the daily price changes
    delta = prices[symbol].diff()

    # Separate the positive and negative gains
    gain = (delta.where(delta > 0, 0))
    loss = (-delta.where(delta < 0, 0))

    # Calculate the average gain and average loss
    avg_gain = gain.rolling(window=window, min_periods=window).mean()
    avg_loss = loss.rolling(window=window, min_periods=window).mean()

    prev_avg_gain = avg_gain.shift(1)
    prev_avg_loss = avg_loss.shift(1)

    # print(prev_avg_gain.head(20), avg_gain.head(20))

    # Calculate the Relative Strength (RS)
    rs = avg_gain / avg_loss
    first = rs.iloc[window - 1]
    rs = (prev_avg_gain * (window-1) + gain) / (prev_avg_loss * (window-1) + loss)
    rs.iloc[window - 1] = first

    # Calculate the Relative Strength Index (RSI)
    prices['RSI'] = 100 - (100 / (1 + rs))
    prices['RSI_oversold'] = 30
    prices['RSI_overbought'] = 70

    return prices

def ema(prices, window=20, symbol='JPM'):
    prices['EMA'] = prices[symbol].ewm(span=window, adjust=False, min_periods=window).mean()
    return prices

def roc(prices, period=12, symbol='JPM'):
    prices['ROC'] = ((prices[symbol] - prices[symbol].shift(period)) / prices[symbol].shift(period)) * 100

    return prices