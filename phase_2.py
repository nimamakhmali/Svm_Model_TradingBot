#Nima
import pandas as pd
import numpy as np
import yfinance as yf
import sklearn.svm as sv
import sklearn.metrics as met
import matplotlib.pyplot as plt
import sklearn.preprocessing as pp
import sklearn.model_selection as ms

class svmTrader:
    def __init__(self,
                 lEMAs:list[int],
                 lSTCs:list[int],
                 TH:float) -> None:
        assert TH > 0, 'TH Must Be A Positive Float Number'
        self.lEMAs = lEMAs
        self.lSCTs = lSTCs
        self.TH    = TH

    def EMA(self,
            DF:pd.DataFrame,
            a:int=1) -> tuple[pd.DataFrame,
                              list[str]]:
        FNs = []
        for lEMA in self.lEMAs:
            Alpha = (1 + a) / (lEMA + a)  
            DF.loc[:, f'EMA({lEMA})'] = DF.loc[:, 'Close'].ewm(alpha=Alpha).mean()  
            DF.loc[:, f'f-EMA({lEMA})'] = DF.loc[:, 'Close'] / DF.loc[:, f'EMA({lEMA})'] - 1
            FNs.append(f'f-EMA({lEMA})')
        return DF, FNs

    def STC(self,
            DF:pd.DataFrame) -> tuple[pd.DataFrame,
                                      list[str]]:
        FNs = []
        for lSTC in self.lSCTs:
            LL = DF.loc[:, 'Low'].rolling(window=lSTC).min()
            HH = DF.loc[:, 'High'].rolling(window=lSTC).max()
            DF.loc[:, f'STC({lSTC})'] = 100 * (DF.loc[:, 'Close'] - LL) / (HH - LL)
            FNs.append(f'STC({lSTC})')
            return DF, FNs

    def ProcessDataset(self,
                       DF:pd.DataFrame) -> tuple[pd.DataFrame,
                                                 list[str],
                                                 list[str]]: 
        DF, FNs1 = self.EMA(DF)
        DF, FNs2 = self.STC(DF)
        FNs = FNs1 + FNs2
        DF.loc[:, 'r'] = DF.loc[:, 'Close'] / DF.loc[:, 'Open'] - 1
        DF.loc[:, 'CC'] = np.where(DF.loc[:, 'r'] < -self.TH / 100, 0, np.nan)
        DF.loc[:, 'CC'] = np.where(DF.loc[:, 'r'] > self.TH / 100, 2, DF.loc[:, 'CC'])
        DF.loc[:, 'CC'].fillna(value=1, inplace=True)
        DF.loc[:, 'r(t+1)'] = DF.loc[:, 'r'].shift(periods=-1)
        DF.loc[:, 'CC(t+1)'] = DF.loc[:, 'CC'].shift(periods=-1)
        TNs = ['CC(t+1)']     
        DF.dropna(axis=0, inplace=True)
        return DF, FNs, TNs
    
                              