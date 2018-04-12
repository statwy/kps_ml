# -*- coding: utf-8 -*-
"""
Created on Wed Apr  4 15:23:03 2018

@author: user
"""
import pandas as pd
import numpy as np
import math
from scipy.stats import norm
from scipy import ndimage
import matplotlib.pyplot as plt
from pandas.plotting import scatter_matrix
import datetime

data=pd.read_csv("data/data_refined.csv")
data=data[['5min_0','K_price','K_volume','K_count','Ex_rate','premium']]
data.index=data['5min_0']

ft =KgetFeatureSet(data, u=0.8, d=-0.7, period=20)
ft.tail(10)

# DATA > KDATA 로부터 Feature Set을 구성한다
def KgetFeatureSet(data, u, d, period):
    
      
    # Feature value를 계산한 후 Z-score Normalization 한다
    macd = scale(MACD(data, 12, 26, 9))
    rsi = scale(RSI(data, 40))
    obv = scale(OBV(data, ext=True))
    #fliquidity = scale(Liquidity(data))
    #fparkinson = scale(ParkinsonVol(data, 10))
    vol = scale(CloseVol(data, 10))
    
    kft = pd.DataFrame()
    kft['kmacd'] = macd
    kft['krsi'] = rsi
    kft['kobv'] = obv
    #kft['liquidity'] = fliquidity
    #kft['parkinson'] = fparkinson
    kft['kvolatility'] = vol
    kft['kclass'] = getUpDnClass(data, u, d, period)
    kft = kft.dropna()
    
    # Feature들의 value (수준) 보다는 방향 (up, down)을 분석하는 것이 의미가 있어 보임.
    # 방향을 어떻게 검출할 지는 향후 연구 과제로 한다

    return kft

# Supervised Learning을 위한 class를 부여한다
# 
# up : 목표 수익률 표준편차
# dn : 손절률 표준편차
# period : holding 기간
# return : 0 - 주가 횡보, 1 - 주가 하락, 2 - 주가 상승
# ---------------------------------------------------
def getUpDnClass(data, up, dn, period):
    # 주가 수익률의 표준편차를 측정한다
    r = []
    for curr, prev in zip(data.itertuples(), data.shift(1).itertuples()):
        if math.isnan(prev.K_price):
            continue
        r.append(np.log(curr.K_price / prev.K_price))
    s = np.std(r)
    
    # 목표 수익률과 손절률을 계산한다
    uLimit = up * s * np.sqrt(period)
    dLimit = dn * s * np.sqrt(period)
    
    # 가상 Trading을 통해 미래 주가 방향에 대한 Class를 결정한다
    rclass = []
    for i in range(len(data) - 1):
        # 매수 포지션을 취한다
        buyPrc = data.iloc[i].K_price
        y = np.nan
            
        # 매수 포지션 이후 청산 지점을 결정한다
        for k in range(i+1, len(data)):
            sellPrc = data.iloc[k].K_price
            
            # 수익률을 계산한다
            rtn = np.log(sellPrc / buyPrc)
            
            # 목표 수익률이나 손절률에 도달하면 루프를 종료한다
            if k > i + period:
                # hoding 기간 동안 목표 수익률이나 손절률에 도달하지 못했음
                # 주가가 횡보한 것임
                y = 0
                break
            else:
                if rtn > uLimit:
                    y = 2       # 수익
                    break
                elif rtn < dLimit:
                    y = 1       # 손실
                    break

        rclass.append(y)
    
    rclass.append(np.nan)
    return pd.DataFrame(rclass, index=data.index)

# MACD 지표를 계산한다
# MACD Line : 12-day EMA - 26-day EMA
# Signal Line : 9-day EMA of MACD line
# MACD oscilator : MACD Line - Signal Line
# ----------------------------------------
def MACD(data, nFast=12, nSlow=26, nSig=9, percent=True):
    ema1 = EMA(data.K_price, nFast)
    ema2 = EMA(data.K_price, nSlow)
    
    if percent:
        macdLine =  100 * (ema1 - ema2) / ema2
    else:
        macdLine =  ema1 - ema2
    signalLine = EMA(macdLine, nSig)
    
    return pd.DataFrame(macdLine - signalLine, index=data.index)





# 지수이동평균을 계산한다
# data : Series
def EMA(data, n):
    ma = []
    
    # data 첫 부분에 na 가 있으면 skip한다
    x = 0
    while True:
        if math.isnan(data[x]):
            ma.append(data[x])
        else:
            break;
        x += 1
        
    # x ~ n - 1 기간까지는 na를 assign 한다
    for i in range(x, x + n - 1):
        ma.append(np.nan)
    
    # x + n - 1 기간은 x ~ x + n - 1 까지의 평균을 적용한다
    sma = np.mean(data[x:(x + n)])
    ma.append(sma)
    
    # x + n 기간 부터는 EMA를 적용한다
    k = 2 / (n + 1)
    
    for i in range(x + n, len(data)):
        #print(i, data[i])
        ma.append(ma[-1] + k * (data[i] - ma[-1]))
    
    return pd.Series(ma, index=data.index)

# RSI 지표를 계산한다. (Momentum indicator)
# U : Gain, D : Loss, AU : Average Gain, AD : Average Loss
# smoothed RS는 고려하지 않았음.
# --------------------------------------------------------
def RSI(data, n=14):
    price = pd.DataFrame(data.K_price)
    U = np.where(price.diff(1) > 0, price.diff(1), 0)
    D = np.where(price.diff(1) < 0, price.diff(1) * (-1), 0)
    
    U = pd.DataFrame(U, index=data.index)
    D = pd.DataFrame(D, index=data.index)
    
    AU = U.rolling(window=n).mean()
    AD = D.rolling(window=n).mean()

    return 100 * AU / (AU + AD)
    
# On Balance Volume (OBV) : buying and selling pressure
# ext = False : 기존의 OBV
# ext = True  : Extended OBV. 가격 변화를 이용하여 거래량을 매수수량, 매도수량으로 분해하여 매집량 누적
# -------------------------------------------------------------------------------------------------
def OBV(data, ext=True):
    obv = [0]
    
    # 기존의 OBV
    if ext == False:
        # 기술적 지표인 OBV를 계산한다
        for curr, prev in zip(data.itertuples(), data.shift(1).itertuples()):
            if math.isnan(prev.K_volume):
                continue
            
            if curr.K_price > prev.K_price:
                obv.append(obv[-1] + curr.K_volume)
            if curr.K_price < prev.K_price:
                obv.append(obv[-1] - curr.K_volume)
            if curr.K_price == prev.K_price:
                obv.append(obv[-1])
    # Extendedd OBV
    else:
        # 가격 변화를 측정한다. 가격 변화 = 금일 종가 - 전일 종가
        deltaPrice = data['K_price'].diff(1)
        deltaPrice = deltaPrice.dropna(axis = 0)
        
        # 가격 변화의 표준편차를 측정한다
        stdev = np.std(deltaPrice)
        
        for curr, prev in zip(data.itertuples(), data.shift(1).itertuples()):
            if math.isnan(prev.K_price):
                continue
            
            buy = curr.K_volume * norm.cdf((curr.K_price - prev.K_price) / stdev)
            sell = curr.K_volume - buy
            bs = abs(buy - sell)
            
            if curr.K_price > prev.K_price:
                obv.append(obv[-1] + bs)
            if curr.K_price < prev.K_price:
                obv.append(obv[-1] - bs)
            if curr.K_price == prev.K_price:
                obv.append(obv[-1])
        
    return pd.DataFrame(obv, index=data.index)

def CloseVol(data, n):
    rtn = pd.DataFrame(data['K_price']).apply(lambda x: np.log(x) - np.log(x.shift(1)))
    vol = pd.DataFrame(rtn).rolling(window=n).std()

    return pd.DataFrame(vol, index=data.index)
    
## 당일의 High price와 Low price를 이용하여 Parkinson 변동성 (장 중 변동성)을 계산한다.
#def ParkinsonVol(data, n):
#    vol = []
#    for i in range(n-1):
#        vol.append(np.nan)
#        
#    for i in range(n-1, len(ohlc)):
#        sigma = 0
#        for k in range(0, n):
#            sigma += np.log(ohlc.iloc[i-k].High / ohlc.iloc[i-k].Low) ** 2
#        vol.append(np.sqrt(sigma / (n * 4 * np.log(2))))
#        
#    return pd.DataFrame(vol, index=ohlc.index)

# Z-score normalization
def scale(data):
    col = data.columns[0]
    return (data[col] - data[col].mean()) / data[col].std()

#시계열을 평활화한다
def smooth(data, s=5):
    y = data[data.columns[1]].values
    #y = data[data.columns[0]].values
    w = np.isnan(y)
    y[w] = 0.
    sm = ndimage.gaussian_filter1d(y, s)
    return pd.DataFrame(sm)

# OHLCV 데이터에서 종가 (Close)를 기준으로 과거 n-기간 동안의 Pattern을 구성한다
#def getClosePattern(data, n):
#    loc = tuple(range(1, len(data) - n, 3))
#    
#    # n개의 column을 가진 데이터프레임을 생성한다
#    column = [str(e) for e in range(1, (n+1))]
#    df = pd.DataFrame(columns=column)
#    
#    for i in loc:       
#        pt = data['Close'].iloc[i:(i+n)].values
#        pt = (pt - pt.mean()) / pt.std()
#        df = df.append(pd.DataFrame([pt],columns=column, index=[data.index[i+n]]), ignore_index=False)
#        
#    return df