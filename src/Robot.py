import os
import numpy as np
import pandas as pd
import yfinance as yf
import sklearn.svm as sv
import sklearn.metrics as met
import matplotlib.pyplot as plt
import sklearn.preprocessing as pp
import sklearn.model_selection as ms

class svmTrder:
    def __init__(self, lEMAs: list[int], lSTCs: list[int], TH: float, ntd: float, rfr: float) -> None:
        assert TH > 0, 'TH Must Be A Positive Float'
        self.lEMAs = lEMAs
        self.lSTCs = lSTCs
        self.TH = TH
        self.ntd = ntd
        self.rfr = rfr

    def EMA(self, DF: pd.DataFrame, a: int = 1) -> tuple[pd.DataFrame, list[str]]:
        FNs = []
        for lEMA in self.lEMAs:
            Alpha = (1 + a) / (lEMA + a)
            EMA_values = DF['Close'].ewm(alpha=Alpha).mean()
            DF[f'EMA({lEMA})'] = EMA_values
            DF[f'f-EMA({lEMA})'] = DF['Close'] / EMA_values - 1
            FNs.append(f'f-EMA({lEMA})')
        return DF, FNs

    def STC(self, DF: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
        FNs = []
        for lSTC in self.lSTCs:
            LL = DF['Low'].rolling(window=lSTC).min()
            HH = DF['High'].rolling(window=lSTC).max()
            DF[f'STC({lSTC})'] = 100 * (DF['Close'] - LL) / (HH - LL)
            FNs.append(f'STC({lSTC})')
        return DF, FNs

    def ProcessDataset(self, DF: pd.DataFrame) -> tuple[pd.DataFrame, list[str], list[str]]:
        DF = DF.copy()
        DF, FNs1 = self.EMA(DF)
        DF, FNs2 = self.STC(DF)
        FNs = FNs1 + FNs2

        DF['r'] = DF['Close'] / DF['Open'] - 1
        DF['CC'] = np.where(DF['r'] < -self.TH / 100, 0, np.nan)
        DF['CC'] = np.where(DF['r'] > self.TH / 100, 2, DF['CC'])
        DF['CC'] = DF['CC'].fillna(value=1)

        DF['r(t+1)'] = DF['r'].shift(periods=-1)
        DF['CC(t+1)'] = DF['CC'].shift(periods=-1)
        TNs = ['CC(t+1)']

        DF.dropna(axis=0, inplace=True)
        return DF, FNs, TNs

    def Fit(self, trDF: pd.DataFrame, vaDF: pd.DataFrame, H: dict) -> None:
        trDF, FNs, TNs = self.ProcessDataset(trDF)
        vaDF, FNs, TNs = self.ProcessDataset(vaDF)

        trX0 = trDF[FNs].to_numpy()
        vaX0 = vaDF[FNs].to_numpy()
        trY = trDF[TNs[0]].to_numpy()
        vaY = vaDF[TNs[0]].to_numpy()

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

        print(f'Best Hyperparameters: {BestP}')
        print(f'Best Validation F1 Score Macro Average: {BestF1ma:.2f} %')
        self.Model = sv.SVC(**BestP, random_state=0)
        self.Model.fit(trX, trY)

    def Sharpe(self, Returns: np.ndarray) -> float:
        EDR = Returns.mean()
        Variance = Returns.var()
        m = self.ntd * EDR - self.rfr / 100
        s = (self.ntd * Variance) ** 0.5
        return m / s

    def Sortino(self, Returns: np.ndarray) -> float:
        EDR = Returns.mean()
        m = self.ntd * EDR - self.rfr / 100
        SemiVariance = np.power(Returns[Returns < EDR] - EDR, 2).mean()
        s = (self.ntd * SemiVariance) ** 0.5
        return m / s

    def Evaluate(self, DF: pd.DataFrame, Dataset: str) -> None:
        DF, FNs, TNs = self.ProcessDataset(DF)
        if len(DF) == 0:
            print(f'[WARNING] Dataset {Dataset} is empty after processing.')
            return

        X0 = DF[FNs].to_numpy()
        Y = DF[TNs[0]].to_numpy()
        X = self.Scaler.transform(X0)
        P = self.Model.predict(X)

        CR = met.classification_report(Y, P, labels=[0, 1, 2], target_names=['Sell', 'Hold', 'Buy'])

        nD = Y.size
        tickerReturns = DF['r(t+1)'].to_numpy()
        modelReturns = tickerReturns * (P - 1)
        tickerEDR = tickerReturns.mean()
        modelEDR = modelReturns.mean()
        tickerACDR = (np.cumprod(tickerReturns + 1)[-1]) ** (1 / nD) - 1
        modelACDR = (np.cumprod(modelReturns + 1)[-1]) ** (1 / nD) - 1
        tickerEYR = self.ntd * tickerEDR
        modelEYR = self.ntd * modelEDR
        tickerACYR = (1 + tickerACDR) ** self.ntd - 1
        modelACYR = (1 + modelACDR) ** self.ntd - 1
        tickerSharpe = self.Sharpe(tickerReturns)
        modelSharpe = self.Sharpe(modelReturns)
        tickerSortino = self.Sortino(tickerReturns)
        modelSortino = self.Sortino(modelReturns)

        print('_' * 60)
        print(f'Model Report On {Dataset} Dataset:')
        print(f'Classification Report:\n{CR}')
        print('_' * 60)
        print(f'Ticker Expected Daily Return: {100 * tickerEDR:.2f} %')
        print(f'Model Expected Daily Return: {100 * modelEDR:.2f} %')
        print(f'Ticker Average Compounded Daily Return: {100 * tickerACDR:.2f} %')
        print(f'Model Average Compounded Daily Return: {100 * modelACDR:.2f} %')
        print(f'Ticker Expected Yearly Return: {100 * tickerEYR:.2f} %')
        print(f'Model Expected Yearly Return: {100 * modelEYR:.2f} %')
        print(f'Ticker Average Compounded Yearly Return: {100 * tickerACYR:.2f} %')
        print(f'Model Average Compounded Yearly Return: {100 * modelACYR:.2f} %')
        print('_' * 60)
        print(f'Ticker Sharpe: {tickerSharpe:.2f}')
        print(f'Model Sharpe: {modelSharpe:.2f}')
        print(f'Ticker Sortino: {tickerSortino:.2f}')
        print(f'Model Sortino: {modelSortino:.2f}')
        print('_' * 60)

    def PlotCumulativeReturns(self, DF: pd.DataFrame, Dataset: str) -> None:
        DF, FNs, _ = self.ProcessDataset(DF)
        if len(DF) == 0:
            print(f'[WARNING] Skipping PlotCumulativeReturns: no data for {Dataset}')
            return
        X0 = DF[FNs].to_numpy()
        X = self.Scaler.transform(X0)
        P = self.Model.predict(X)

        tickerReturns = 100 * DF['r(t+1)'].to_numpy()
        modelReturns = tickerReturns * (P - 1)
        tickerCumulativeReturns = np.cumsum(tickerReturns)
        modelCumulativeReturns = np.cumsum(modelReturns)

        plt.plot(DF.index, tickerCumulativeReturns, ls='-', lw=1, c='teal', label='Ticker')
        plt.plot(DF.index, modelCumulativeReturns, ls='-', lw=1, c='crimson', label='Model')
        plt.title(f'Cumulative Summation Of Returns For {Dataset} Dataset')
        plt.xlabel('Date')
        plt.ylabel('Return (%)')
        plt.legend()
        plt.show()

    def PlotTrades(self, DF: pd.DataFrame, Dataset: str) -> None:
        DF, FNs, _ = self.ProcessDataset(DF)
        if len(DF) == 0:
            print(f'[WARNING] Skipping PlotTrades: no data for {Dataset}')
            return
        X0 = DF[FNs].to_numpy()
        X = self.Scaler.transform(X0)
        P = self.Model.predict(X)

        sMask = P == 0
        hMask = P == 1
        bMask = P == 2

        plt.plot(DF.index, DF['Close'], ls='-', lw=0.8, c='k', label='Close')
        plt.scatter(DF.index[sMask], DF.loc[sMask, 'Close'], s=40, c='r', alpha=0.8, label='Sell')
        plt.scatter(DF.index[hMask], DF.loc[hMask, 'Close'], s=20, c='gray', alpha=0.8, label='Hold')
        plt.scatter(DF.index[bMask], DF.loc[bMask, 'Close'], s=40, c='g', alpha=0.8, label='Buy')

        for lEMA in self.lEMAs:
            plt.plot(DF.index, DF[f'EMA({lEMA})'], ls='--', lw=1.2, label=f'EMA({lEMA})')

        plt.title(f'Model Trades History For {Dataset} Dataset')
        plt.xlabel('Date')
        plt.ylabel('Price')
        plt.yscale('log')
        plt.legend()
        plt.show()

def Fetch(Ticker: str, Start: str, End: str) -> pd.DataFrame:
    if not os.path.exists('Data'):
        os.mkdir('Data')
    SavePath = f'Data/{Ticker}-{Start}-{End}.csv'
    if not os.path.exists(SavePath):
        DF = yf.download(tickers=Ticker, start=Start, end=End, interval='1d')
        if len(DF) > 0:
            DF.to_csv(SavePath, sep=',', index=True, index_label='Date', encoding='UTF-8')
    else:
        DF = pd.read_csv(SavePath, sep=',', header=0, index_col='Date', encoding='UTF-8')
        DF.index = pd.to_datetime(DF.index)
    return DF

trDF = Fetch('BTC-USD', '2017-01-01', '2020-01-01')
vaDF = Fetch('BTC-USD', '2020-01-01', '2024-01-01')
teDF = Fetch('BTC-USD', '2022-01-01', '2025-07-19')

lEMAs = [15, 35]
lSTCs = [10, 21]
TH = 0.8
ntd = 252.03
rfr = 10

H = {
    'C': [1.1],
    'kernel': ['poly'],
    'degree': [3, 4],
    'gamma': ['scale'],
    'coef0': [+0.2, +0.3],
    'class_weight': ['balanced'],
    'max_iter': [-1, 700]
}

Trader = svmTrder(lEMAs, lSTCs, TH, ntd, rfr)
Trader.Fit(trDF, vaDF, H)
Trader.Evaluate(teDF, 'Test')
Trader.PlotCumulativeReturns(teDF, 'Test')
Trader.PlotTrades(teDF, 'Test')
