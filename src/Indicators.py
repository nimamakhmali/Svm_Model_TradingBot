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

def EMA(DF:pd.DataFrame, L:int, a:int=1) -> pd.DataFrame:
    Alpha = (1 + a) / (L + a) # Calculating Alpha
    DF.loc[:, f'EMA({L})'] = DF.loc[:, 'Close'].ewm(alpha=Alpha).mean()
    return DF

def STC(DF:pd.DataFrame, L:int) -> pd.DataFrame:
    LL = DF.loc[:, 'Low'].rolling(window=L).min() # Moving Min
    HH = DF.loc[:, 'High'].rolling(window=L).max() # Moving Max
    DF.loc[:, f'STC({L})'] = 100 * (DF.loc[:, 'Close'] - LL) / (HH - LL)
    return DF


DF = Fetch('MSFT', '2020-01-01', '2021-01-01')

DF = EMA(DF, 20)
DF = EMA(DF, 50)

plt.plot(DF.loc[:, 'Close'], ls='-', lw=0.9, c='k', label='Close')
plt.plot(DF.loc[:, 'EMA(20)'], ls='--', lw=1.1, c='crimson', label='EMA(20)')
plt.plot(DF.loc[:, 'EMA(50)'], ls='--', lw=1.1, c='teal', label='EMA(50)')
plt.title('MSFT Share Price Over Time')
plt.xlabel('Date')
plt.ylabel('Price ($)')
plt.legend()
plt.show()

DF = STC(DF, 30)

plt.subplot(3, 1, (1, 2))
plt.plot(DF.loc[:, 'Close'], ls='-', lw=0.9, c='k', label='Close')
plt.xlim(left=DF.index[0], right=DF.index[-1])
plt.legend()

plt.subplot(3, 1, 3)
plt.plot(DF.loc[:, 'STC(30)'], ls='-', lw=1, c='crimson', label='STC(30)')
plt.axhline(y=80, ls='-', lw=0.9, c='gray')
plt.axhline(y=50, ls='-', lw=0.9, c='gray')
plt.axhline(y=20, ls='-', lw=0.9, c='gray')
plt.xlim(left=DF.index[0], right=DF.index[-1])
plt.legend()

plt.show()