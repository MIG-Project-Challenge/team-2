from collections import defaultdict
import numpy as np
from pathlib import Path
import argparse
import talib as ta
import glob
import pandas as pd
import matplotlib.pyplot as plt

# read the data from the CSV files
train_data = pd.read_csv('train_data_50.csv') 

# change date into datetime objects
train_data['Date'] = pd.to_datetime(train_data['Date'])

# set indexes 
train_data.set_index(["Ticker", "Date"], inplace=True)

# sort data by tickers
tickers = sorted(train_data.index.get_level_values('Ticker').unique())

# array of all stocks and close prices for RSI
open_prices = []

for ticker in tickers:
    stock_close_data = train_data.loc[ticker]["Open"]
    open_prices.append(stock_close_data.values)

open_prices = np.stack(open_prices)
print(open_prices.shape)
print(open_prices)

# array for final return - trades
trades = np.zeros_like(open_prices)
trades

# actually calculate RSI value
# if rsi > 70 the stock is overbought, if rsi < 30 the stock is underbought
for stock in range(len(open_prices)):
    talib_rsi = ta.RSI(stock-stock_close_data, timeperiod=40)

    for day in range(1, len(open_prices[0])-1):
        if talib_rsi[day] > 70: # buy
            trades[stock][day+1] = -1
        elif talib_rsi[day] < 30: # sell/short
            trades[stock][day+1] = 1
        else: # do nothing
            trades[stock][day+1] = 0