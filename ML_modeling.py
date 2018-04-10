# -*- coding: utf-8 -*-
import pandas as pd 
import matplotlib.pyplot as plt
import numpy as np
#from pandas.plotting import scatter_matrix
import datetime
import math
#from scipy.stats import norm
#from scipy import ndimage
#from sklearn.model_selection import train_test_split
#from sklearn.neighbors import KNeighborsClassifier


def bollingerband(x,y=100,sigma=2) :
    x=pd.DataFrame(x)
    mva=x.rolling(window=y).mean()
    mvstd=x.rolling(window=y).std()
    upper_bound=mva+sigma*mvstd
    lower_bound=mva-sigma*mvstd
    return upper_bound, lower_bound


def seasonality (x) :
    if 8<pd.to_datetime(x).hour<24 :
        return 1
    else :
        return 0
    

def RSI(ohlc, n=14):
    closePrice = pd.DataFrame(ohlc)
    U = np.where(closePrice.diff(1) > 0, closePrice.diff(1), 0)
    D = np.where(closePrice.diff(1) < 0, closePrice.diff(1) * (-1), 0)
    
    U = pd.DataFrame(U, index=ohlc.index)
    D = pd.DataFrame(D, index=ohlc.index)
    
    AU = U.rolling(window=n).mean()
    AD = D.rolling(window=n).mean()

    return 100 * AU / (AU + AD)


data=pd.read_csv("data/data_refined.csv")

data_hy_k=pd.read_csv("data/kr_hy.csv")
data_hy_u=pd.read_csv("data/us_hy.csv")

data1=pd.merge(data_hy_k, data_hy_u, left_on='5min_0', right_on='5min_0')
data2=pd.merge(data, data1, left_on='5min_0', right_on='5min_0')
data=data2
data.index=data['day']
#data=data[['day','5min_0','K_price','K_volume','K_count','U_price','Adj_U_price','U_volume','U_count','Ex_rate','premium']]
data['RSI_k']=RSI(data.K_price)
data['RSI_u']=RSI(data.U_price)
data['season']=list(map(seasonality , list(data['5min_0']) ) )
data['adjusted_volume']=data['K_volume']*data['RSI_k']
data['bollinger_k_upper'],data['bollinger_k_lower']=bollingerband(data['K_price'],100,2)
data['bollinger_u_upper'],data['bollinger_u_lower']=bollingerband(data['U_price'],100,2)
data['bollinger_price_k']=(data['K_price']-data['bollinger_k_lower'])/(data['bollinger_k_upper']-data['bollinger_k_lower'])
data['bollinger_price_u']=(data['U_price']-data['bollinger_u_lower'])/(data['bollinger_u_upper']-data['bollinger_u_lower'])
data['bollinger_price_k_and_u']=data['bollinger_price_k']-data['bollinger_price_u']

data=data.dropna()

#a=getUpDnClass(data_for_knn, 0.8, -0.8,20)

#data.describe()
#data.corr()
#plt.plot(data['premium'])
#data.corr()
#data['premium'].hist()
#scatter_matrix(data[['premium','K_price']], figsize=(10,8), diagonal='kde', color='brown', marker='o', s=2.5)
#plt.show()


#data_explor=data.loc['2018-01-30 18:20:00':'2018-02-20 23:55:00']
#data_explor.index=data_explor['5min_0']
#data_explor.corr()
#plt.plot(data_explor.RSI_k)
#plt.plot(data_explor.RSI_u)

#data=data.loc['2017-04-01 00:00:00':'2018-03-13 23:55:00']
#
#data_for_knn=data[['day','5min_0', 'K_price', 'K_volume', 'K_count', 'U_price','U_volume', 'U_count', 'Ex_rate','RSI_k', 'RSI_u', 'season',
#       'adjusted_volume','bollinger_price_k','bollinger_price_u','bollinger_price_k_and_u','premium']]
#
#


data['pre_class']=0
#pre_class={'class':[]}
for i in range(0,97422) :
    try :
        nextday=pd.to_datetime(data.iloc[i]['day'])+datetime.timedelta(days=1)
        nextday=nextday.strftime("%Y-%m-%d")
        mm=data.loc[nextday]['premium'].min()
        mx=data.loc[nextday]['premium'].max()
       
        min_val=(100-data.iloc[i]['premium'])*(1+(mm)/100)
        max_val=(100-data.iloc[i]['premium'])*(1+(mx)/100)
        temp=0
        if min_val<97 :
            temp=1
        if max_val>103 :
            temp=2
        if min_val<97 and max_val>103 :
            temp=0
        data['pre_class'].iloc[i]=temp
        print(min_val," ",max_val, "temp:", temp)
    except :
        data['pre_class'].iloc[i]='Nan'

data=data['2017-04-01 00:00:00':'2018-03-12 23:55:00']
data.to_csv("data/knndata.csv") 










#
#data_for_knn['pre_class']=0
##pre_class={'class':[]}
#for i in range(0,97422) :
#    try :
#        nextday=pd.to_datetime(data_for_knn.iloc[i]['day'])+datetime.timedelta(days=1)
#        nextday=nextday.strftime("%Y-%m-%d")
#        mm=data_for_knn.loc[nextday]['premium'].min()
#        mx=data_for_knn.loc[nextday]['premium'].max()
#       
#        min_val=(100-data_for_knn.iloc[i]['premium'])*(1+(mm)/100)
#        max_val=(100-data_for_knn.iloc[i]['premium'])*(1+(mx)/100)
#        temp=0
#        if min_val<97 :
#            temp=1
#        if max_val>103 :
#            temp=2
#        if min_val<97 and max_val>103 :
#            temp=0
#        data_for_knn['pre_class'].iloc[i]=temp
#        print(min_val," ",max_val, "temp:", temp)
#    except :
#        data_for_knn['pre_class'].iloc[i]='Nan'
#
#data_for_knn=data_for_knn['2017-04-01 00:00:00':'2018-03-12 23:55:00']
#data_for_knn.to_csv("data/knndata.csv") 

#    data11=data_for_knn['day']==nextday    
#    data_for_knn['flag']=data11
#    data_for_knn['premium'].groupby(data_for_knn['flag']).max().loc[True]
#    data_for_knn['premium'].groupby(data_for_knn['flag']).min().loc[True]
        

#plt.hist(data_for_knn['pre_class'])
#plt.show()


# Train 데이터 세트와 Test 데이터 세트를 구성한다
x = data_for_knn.iloc[:, 0:6]
y = data_for_knn['class']
trainX, testX, trainY, testY = train_test_split(x, y, test_size = 0.2, random_state=None)

# KNN 으로 Train 데이터 세트를 학습한다.
knn = KNeighborsClassifier(n_neighbors=50, p=2, metric='minkowski')
knn.fit(trainX, trainY)

# Test 세트의 Feature에 대한 class를 추정하고, 정확도를 계산한다
predY = knn.predict(testX)
accuracy = 100 * (testY == predY).sum() / len(predY)
print()
print("* 시험용 데이터로 측정한 정확도 = %.2f" % accuracy, '%')

# Train 세트의 Feature에 대한 class를 추정하고, 정확도를 계산한다
predY = knn.predict(trainX)
accuracy = 100 * (trainY == predY).sum() / len(predY)
print("* 학습용 데이터로 측정한 정확도 = %.2f" % accuracy, '%')

# k를 변화시켜가면서 정확도를 측정해 본다
testAcc = []
trainAcc = []
for k in range(5, 100):
    # KNN 으로 Train 데이터 세트를 학습한다.
    knn = KNeighborsClassifier(n_neighbors=k, p=2, metric='minkowski')
    knn.fit(trainX, trainY)
    
    # Test 세트의 Feature에 대한 정확도
    predY = knn.predict(testX)
    testAcc.append((testY == predY).sum() / len(predY))
    
    # Train 세트의 Feature에 대한 정확도
    predY = knn.predict(trainX)
    trainAcc.append((trainY == predY).sum() / len(predY))

plt.figure(figsize=(8, 5))
plt.plot(testAcc, label="Test Data")
plt.plot(trainAcc, label="Train Data")
plt.legend()
plt.xlabel("k")
plt.ylabel("Accuracy")
plt.show()




#
## Supervised Learning을 위한 class를 부여한다
## 
## up : 목표 수익률 표준편차
## dn : 손절률 표준편차
## period : holding 기간
## return : 0 - 주가 횡보, 1 - 주가 하락, 2 - 주가 상승
## ---------------------------------------------------
#def getUpDnClass(data, up, dn, period):
#    # 주가 수익률의 표준편차를 측정한다
#    data=data_for_knn
#    up=0.8
#    dn=-0.8
#    period=100
#    r = []
#    for curr, prev in zip(data.itertuples(), data.shift(1).itertuples()):
#        if math.isnan(prev.premium):
#            continue
#        r.append(curr.premium - prev.premium)
#    s = np.std(r)
#    
#    # 목표 수익률과 손절률을 계산한다
#    uLimit = up * s * np.sqrt(period)
#    dLimit = dn * s * np.sqrt(period)
#    
#    # 가상 Trading을 통해 미래 주가 방향에 대한 Class를 결정한다
#    rclass = []
#    for i in range(len(data) - 1):
#        # 매수 포지션을 취한다
#        buyPrc = data.iloc[i].premium
#        y = np.nan
#            
#        # 매수 포지션 이후 청산 지점을 결정한다
#        for k in range(i+1, len(data)):
#            sellPrc = data.iloc[k].premium
#            
#            # 수익률을 계산한다
#            rtn = sellPrc-buyPrc
#            
#            # 목표 수익률이나 손절률에 도달하면 루프를 종료한다
#            if k > i + period:
#                # hoding 기간 동안 목표 수익률이나 손절률에 도달하지 못했음
#                # 주가가 횡보한 것임
#                y = 0
#                break
#            else:
#                if rtn > uLimit:
#                    y = 2       # 수익
#                    break
#                elif rtn < dLimit:
#                    y = 1       # 손실
#                    break
#
#        rclass.append(y)
#    
#    rclass.append(np.nan)
#    return pd.DataFrame(rclass, index=data.index)
#
#
#
# ######################################################################################################################
#########################################################################################################################
#
#data=pd.read_csv("data/data_refined.csv")
#data=data[['5min_0','K_price','K_volume','K_count','Ex_rate','premium']]
#data.index=data['5min_0']
#
#ft =getFeatureSet(data, u=0.8, d=-0.7, period=20)
#ft.tail(10)
#
## DATA > KDATA 로부터 Feature Set을 구성한다
#def getFeatureSet(data, u, d, period):
#    
#      
#    # Feature value를 계산한 후 Z-score Normalization 한다
#    macd = scale(MACD(data, 12, 26, 9))
#    rsi = scale(RSI(data, 40))
#    obv = scale(OBV(data, ext=True))
#    #fliquidity = scale(Liquidity(data))
#    #fparkinson = scale(ParkinsonVol(data, 10))
#    vol = scale(CloseVol(data, 10))
#    
#    kft = pd.DataFrame()
#    kft['macd'] = macd
#    kft['rsi'] = rsi
#    kft['obv'] = obv
#    #kft['liquidity'] = fliquidity
#    #kft['parkinson'] = fparkinson
#    kft['volatility'] = vol
#    kft['class'] = getUpDnClass(data, u, d, period)
#    kft = kft.dropna()
#    
#    # Feature들의 value (수준) 보다는 방향 (up, down)을 분석하는 것이 의미가 있어 보임.
#    # 방향을 어떻게 검출할 지는 향후 연구 과제로 한다
#
#    return kft
#
#
#
## MACD 지표를 계산한다
## MACD Line : 12-day EMA - 26-day EMA
## Signal Line : 9-day EMA of MACD line
## MACD oscilator : MACD Line - Signal Line
## ----------------------------------------
#def MACD(data, nFast=12, nSlow=26, nSig=9, percent=True):
#    ema1 = EMA(data.K_price, nFast)
#    ema2 = EMA(data.K_price, nSlow)
#    
#    if percent:
#        macdLine =  100 * (ema1 - ema2) / ema2
#    else:
#        macdLine =  ema1 - ema2
#    signalLine = EMA(macdLine, nSig)
#    
#    return pd.DataFrame(macdLine - signalLine, index=data.index)
#
#
#
#
#
## 지수이동평균을 계산한다
## data : Series
#def EMA(data, n):
#    ma = []
#    
#    # data 첫 부분에 na 가 있으면 skip한다
#    x = 0
#    while True:
#        if math.isnan(data[x]):
#            ma.append(data[x])
#        else:
#            break;
#        x += 1
#        
#    # x ~ n - 1 기간까지는 na를 assign 한다
#    for i in range(x, x + n - 1):
#        ma.append(np.nan)
#    
#    # x + n - 1 기간은 x ~ x + n - 1 까지의 평균을 적용한다
#    sma = np.mean(data[x:(x + n)])
#    ma.append(sma)
#    
#    # x + n 기간 부터는 EMA를 적용한다
#    k = 2 / (n + 1)
#    
#    for i in range(x + n, len(data)):
#        #print(i, data[i])
#        ma.append(ma[-1] + k * (data[i] - ma[-1]))
#    
#    return pd.Series(ma, index=data.index)
#
## RSI 지표를 계산한다. (Momentum indicator)
## U : Gain, D : Loss, AU : Average Gain, AD : Average Loss
## smoothed RS는 고려하지 않았음.
## --------------------------------------------------------
#def RSI(data, n=14):
#    price = pd.DataFrame(data.K_price)
#    U = np.where(price.diff(1) > 0, price.diff(1), 0)
#    D = np.where(price.diff(1) < 0, price.diff(1) * (-1), 0)
#    
#    U = pd.DataFrame(U, index=data.index)
#    D = pd.DataFrame(D, index=data.index)
#    
#    AU = U.rolling(window=n).mean()
#    AD = D.rolling(window=n).mean()
#
#    return 100 * AU / (AU + AD)
#    
## On Balance Volume (OBV) : buying and selling pressure
## ext = False : 기존의 OBV
## ext = True  : Extended OBV. 가격 변화를 이용하여 거래량을 매수수량, 매도수량으로 분해하여 매집량 누적
## -------------------------------------------------------------------------------------------------
#def OBV(data, ext=True):
#    obv = [0]
#    
#    # 기존의 OBV
#    if ext == False:
#        # 기술적 지표인 OBV를 계산한다
#        for curr, prev in zip(data.itertuples(), data.shift(1).itertuples()):
#            if math.isnan(prev.K_volume):
#                continue
#            
#            if curr.K_price > prev.K_price:
#                obv.append(obv[-1] + curr.K_volume)
#            if curr.K_price < prev.K_price:
#                obv.append(obv[-1] - curr.K_volume)
#            if curr.K_price == prev.K_price:
#                obv.append(obv[-1])
#    # Extendedd OBV
#    else:
#        # 가격 변화를 측정한다. 가격 변화 = 금일 종가 - 전일 종가
#        deltaPrice = data['K_price'].diff(1)
#        deltaPrice = deltaPrice.dropna(axis = 0)
#        
#        # 가격 변화의 표준편차를 측정한다
#        stdev = np.std(deltaPrice)
#        
#        for curr, prev in zip(data.itertuples(), data.shift(1).itertuples()):
#            if math.isnan(prev.K_price):
#                continue
#            
#            buy = curr.K_volume * norm.cdf((curr.K_price - prev.K_price) / stdev)
#            sell = curr.K_volume - buy
#            bs = abs(buy - sell)
#            
#            if curr.K_price > prev.K_price:
#                obv.append(obv[-1] + bs)
#            if curr.K_price < prev.K_price:
#                obv.append(obv[-1] - bs)
#            if curr.K_price == prev.K_price:
#                obv.append(obv[-1])
#        
#    return pd.DataFrame(obv, index=data.index)
#
#def CloseVol(data, n):
#    rtn = pd.DataFrame(data['K_price']).apply(lambda x: np.log(x) - np.log(x.shift(1)))
#    vol = pd.DataFrame(rtn).rolling(window=n).std()
#
#    return pd.DataFrame(vol, index=data.index)
#    
### 당일의 High price와 Low price를 이용하여 Parkinson 변동성 (장 중 변동성)을 계산한다.
##def ParkinsonVol(data, n):
##    vol = []
##    for i in range(n-1):
##        vol.append(np.nan)
##        
##    for i in range(n-1, len(ohlc)):
##        sigma = 0
##        for k in range(0, n):
##            sigma += np.log(ohlc.iloc[i-k].High / ohlc.iloc[i-k].Low) ** 2
##        vol.append(np.sqrt(sigma / (n * 4 * np.log(2))))
##        
##    return pd.DataFrame(vol, index=ohlc.index)
#
## Z-score normalization
#def scale(data):
#    col = data.columns[0]
#    return (data[col] - data[col].mean()) / data[col].std()
#
##시계열을 평활화한다
#def smooth(data, s=5):
#    y = data[data.columns[1]].values
#    #y = data[data.columns[0]].values
#    w = np.isnan(y)
#    y[w] = 0.
#    sm = ndimage.gaussian_filter1d(y, s)
#    return pd.DataFrame(sm)
#
## OHLCV 데이터에서 종가 (Close)를 기준으로 과거 n-기간 동안의 Pattern을 구성한다
##def getClosePattern(data, n):
##    loc = tuple(range(1, len(data) - n, 3))
##    
##    # n개의 column을 가진 데이터프레임을 생성한다
##    column = [str(e) for e in range(1, (n+1))]
##    df = pd.DataFrame(columns=column)
##    
##    for i in loc:       
##        pt = data['Close'].iloc[i:(i+n)].values
##        pt = (pt - pt.mean()) / pt.std()
##        df = df.append(pd.DataFrame([pt],columns=column, index=[data.index[i+n]]), ignore_index=False)
##        
##    return df   


