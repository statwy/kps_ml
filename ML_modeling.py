# -*- coding: utf-8 -*-
import pandas as pd 
import matplotlib.pyplot as plt
import numpy as np
from pandas.plotting import scatter_matrix
import datetime


data=pd.read_csv("data/data_refined.csv")
data.index=data['day']
data=data[['5min_0','K_price','K_volume','K_count','U_price','Adj_U_price','U_volume','U_count','Ex_rate','premium']]
data['RSI_k']=RSI(data.K_price)
data['RSI_u']=RSI(data.U_price)
data['season']=list(map(seasonality , list(data['5min_0']) ) )
data['adjusted_volume']=data['K_volume']*data['RSI_k']
data['bollinger_k_upper'],data['bollinger_k_lower']=bollingerband(data['K_price'],100,2)
data['bollinger_u_upper'],data['bollinger_u_lower']=bollingerband(data['U_price'],100,2)
data['bollinger_price_k']=(data['K_price']-data['bollinger_k_lower'])/(data['bollinger_k_upper']-data['bollinger_k_lower'])
data['bollinger_price_u']=(data['K_price']-data['bollinger_u_lower'])/(data['bollinger_u_upper']-data['bollinger_u_lower'])
data['bollinger_price_k_and_u']=data['bollinger_price_k']-data['bollinger_price_u']

data.describe()
data.corr()
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


data_test=data.loc['2018-01-08 18:20:00':'2018-03-13 23:55:00']
data_test.corr()


bollingerband(data['K_price'],100,2)





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

    


