import os as os
import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt

def Fetch(Ticker:str, Start:str, End:str) -> pd.DataFrame:
    if not os.path.exists(path='Data'):
        os.mkdir(path='Data')
    SavePath = f'Data/{Ticker}-{Start}-{End}.csv'
    if not os.path.exists(path=SavePath):
        DF = yf.download(tickers=Ticker,
                         start=Start,
                         end=End,
                         interval='1d')
        if len(DF) > 0:
            DF.to_csv(path_or_buf=SavePath,
                      sep=',',
                      index=True,
                      index_label='Date',
                      encoding='UTF-8')
    else:
        DF = pd.read_csv(filepath_or_buffer=SavePath,
                         sep=',',
                         header=0,
                         index_col='Date',
                         encoding='UTF-8')
        DF.index = pd.to_datetime(DF.index)
    return DF

DF = Fetch('MSFT', '2020-01-01', '2021-01-01')

plt.plot(DF.loc[:, 'Close'])
plt.title('MSFT Share Price Over Time')
plt.xlabel('Date')
plt.ylabel('Price ($)')
plt.show()