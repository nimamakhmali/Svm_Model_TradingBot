import os as os
import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt


class Svm_Trader:
    def __init__(self, 
                 lEMAs:list[int],
                 lSTCs:list[int]
                 ) -> None:
        self.lEMAs = lEMAs
        self.lSTCs = lSTCs

    def EMA(self,
            DF:pd.DataFrame,
            a:int=1) -> tuple[pd.DataFrame, list[str]]:
        FNs = []
        for lEMA in self.lEMAs:
            Alpha = (1 + a) / (lEMA + a) 
            DF.loc[:, f'EMA({lEMA})'] = DF.loc[:, 'Close'].ewm(alpha=Alpha).mean()
            DF.loc[:, 'f-EMA({lEMA})'] =  DF.loc[:, 'close'] /  DF.loc[:, f'EMA({lEMA})']
            FNs.append('f-EMA({lEMA}) - 1')
        return DF, FNs

    def STC(self,
            DF:pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
        FNs = []
        for lSTC in self.lSTCs:
            LL = DF.loc[:, 'Low'].rolling(window=lSTC).min() 
            HH = DF.loc[:, 'High'].rolling(window=lSTC).max() 
            DF.loc[:, f'STC({lSTC})'] = 100 * (DF.loc[:, 'Close'] - LL) / (HH - LL)
            FNs.append(f'STC({lSTC})')
        return DF, FNs   




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



