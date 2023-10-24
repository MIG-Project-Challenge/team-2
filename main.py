from collections import defaultdict
import numpy as np
from pathlib import Path
import argparse
import talib as ta
import glob
import pandas as pd
import matplotlib.pyplot as plt


train_data = pd.read_csv('train_data_50.csv')

# change date into datetime objects
train_data['Date'] = pd.to_datetime(train_data['Date'])

# set indexes 
train_data.set_index(["Ticker", "Date"], inplace=True)
tickers = sorted(train_data.index.get_level_values('Ticker').unique())

open_prices = []

for ticker in tickers:
    stock_close_data = train_data.loc[ticker]["Open"]
    open_prices.append(stock_close_data.values)

open_prices = np.stack(open_prices)
print(open_prices.shape)
print(open_prices)

trades = np.zeros_like(open_prices)
trades

for stock in range(len(open_prices)): 
    fast_sma = ta.SMA(open_prices[stock], timeperiod=5)
    slow_sma = ta.SMA(open_prices[stock], timeperiod=40)

    for day in range(1, len(open_prices[0])-1):
        
        # Buy: fast SMA crosses above slow SMA
        if fast_sma[day] > slow_sma[day] and fast_sma[day-1] <= slow_sma[day-1]:
            # we are trading the next day's open price
            trades[stock][day+1] = 1
        
        # Sell/short: fast SMA crosses below slow SMA
        elif fast_sma[day] < slow_sma[day] and fast_sma[day-1] >= slow_sma[day-1]:
            # we are trading the next day's open price
            trades[stock][day+1] = -1
        # else do nothing
        else:
            trades[stock][day+1] = 0


for ticker in tickers:
    n = ta.AD(train_data.loc[ticker]["High"], train_data.loc[ticker]["Low"], train_data.loc[ticker]["Close"], train_data.loc[ticker]["Volume"])
    plt.plot(n.index,n.values, label=f'{ticker}')

plt.xlabel('Date')
plt.ylabel('Price')
plt.title('Stock Prices')
plt.legend()
plt.show()