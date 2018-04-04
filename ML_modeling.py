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

data.describe()
data.corr()
#plt.plot(data['premium'])
#data.corr()
#data['premium'].hist()
#scatter_matrix(data[['premium','K_price']], figsize=(10,8), diagonal='kde', color='brown', marker='o', s=2.5)
#plt.show()

data_test=data.loc['2018-01-08 18:20:00':'2018-03-13 23:55:00']

data_test.corr()






#data_explor=data.loc['2018-01-30 18:20:00':'2018-02-20 23:55:00']
#data_explor.index=data_explor['5min_0']
#data_explor.corr()
#plt.plot(data_explor.RSI_k)
#plt.plot(data_explor.RSI_u)









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
    
    


