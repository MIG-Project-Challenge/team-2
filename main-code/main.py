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

cash = 25000 #initialize a cash variable to keep track of our money after trades

# calculate golden cross value using TA-lib
# calculate rsi
#use both in tandem

for stock in range(len(open_prices)): 
    fast_sma = ta.SMA(open_prices[stock], timeperiod=5)
    slow_sma = ta.SMA(open_prices[stock], timeperiod=20)
    
    talib_rsi = ta.RSI(stock-stock_close_data, timeperiod=25)

    for day in range(1, len(open_prices[0])-1):
        
        # Buy: fast SMA crosses above slow SMA
        #and rsi is less than 30 -- underbought stock
        if fast_sma[day] > slow_sma[day] and fast_sma[day-1] <= slow_sma[day-1] and talib_rsi[day] < 30:
            # we are trading the next day's open price
            
            if (cash >= open_prices[stock][day] * 30): #if we have enough cash to make 30 trades
                trades[stock][day+1] = 30 #BUY
                cash -= open_prices[stock][day] * 30 #reduce cash accordingly

        # Sell/short: fast SMA crosses below slow SMA
        #and rsi is greater than 70 -- overbought stock
        elif fast_sma[day] < slow_sma[day] and fast_sma[day-1] >= slow_sma[day-1] and talib_rsi[day] > 70:
            # we are trading the next day's open price
        
            trades_stock = trades[stock] 
            numStocks = np.sum(trades_stock) #total stocks we currently have for that stock
            if (numStocks >= 22): #if we can sell 22
                trades[stock][day + 1] = -22 #SELL
                cash += open_prices[stock][day] * 22 #increase cash accordingly
    
        # else do nothing
        else:
            trades[stock][day+1] = 0
