#Nima
import pandas as pd
import numpy as np
import yfinance as yf
import sklearn.svm as sv
import sklearn.metrics as met
import matplotlib.pyplot as plt
import sklearn.preprocessing as pp
import sklearn.model_selection as ms
import os

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

    def Fit(self,
            trDF:pd.DataFrame,
            vaDF:pd.DataFrame,
            H:dict) -> None:
        trDF, FNs, TNs = self.ProcessDataset(trDF)
        vaDF, FNs, TNs = self.ProcessDataset(vaDF)
        trX0 = trDF.loc[:, FNs].to_numpy()
        vaX0 = vaDF.loc[:, FNs].to_numpy()
        trY = trDF.loc[:, TNs[0]].to_numpy()
        vaY = vaDF.loc[:, TNs[0]].to_numpy()
        self.Scaler = pp.StandardScaler()
        trX = self.Scaler.fit_transform(trX0)
        vaX = self.Scaler.transform(vaX0)
        BestP = None
        BestF1ma = 0
        for P in ms.ParameterGrid(H):
            Model = sv.SVC(**P, random_state=0)
            Model.fit(trX, trY)
            vaP = Model.predict(vaX)
            vaF1ma = 100 * met.f1_score(vaY, vaP, average='macro')
            print(f'{P}: {vaF1ma:.2f} %')
            if vaF1ma > BestF1ma:
                BestP = P
                BestF1ma = vaF1ma
        print(f'Best Hyperparameters: {P}')
        print(f'Best Validation F1 Score Macro Average: {BestF1ma:.2f} %')
        self.Model = sv.SVC(**BestP, random_state=0)
        self.Model.fit(trX, trY)



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


trDF = Fetch('TSLA')
vaDF = Fetch('TSLA')

lEMAs = [15]
lSTCs = [21]
TH = 0.008
H = {'C': [0.9, 1, 1.1],
     'kernel': ['rbf', 'linear', 'poly'],
     'gamma': ['scale'],
     'coef0': [-0.2, 0, +0.2],
     'class_weight': ['balanced'],
     'max_iter': [-1, 500, 700]}

Trader = svmTrader(lEMAs, lSTCs, TH)
Trader.Fit(trDF, vaDF, H)
