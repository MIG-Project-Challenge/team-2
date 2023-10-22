import numpy as np
import pandas as pd
import time
import matplotlib.pyplot as plt

train_data = pd.read_csv('train_data_50.csv')

# change date into datetime objects
train_data['Date'] = pd.to_datetime(train_data['Date'])

# set indexes 
train_data.set_index(["Ticker", "Date"], inplace=True)

print(train_data)

tickers = sorted(train_data.index.get_level_values('Ticker').unique())

plt.figure(figsize=(8, 6))

for ticker in tickers:
    stock_close_data = train_data.loc[ticker]["Close"]
    plt.plot(stock_close_data.index,stock_close_data.values, label=f'{ticker}')

plt.xlabel('Date')
plt.ylabel('Price')
plt.title('Stock Prices')
plt.legend()
plt.show()