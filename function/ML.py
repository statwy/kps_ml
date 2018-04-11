# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd


def realDataSet(data, prior=1):
    x= []
    for i in range(len(data)-prior+1):
        a = data[i:(i+prior)]
        x.append(a)
    trainX = np.array(x) 
    return trainX



def getClosePattern(data, n):
    loc = tuple(range(1, len(data) - n, 10))    
    column = [str(e) for e in range(1, (n+1))]
    df = pd.DataFrame(columns=column)    
    for i in loc:       
        pt = data['premium'].iloc[i:(i+n)].values
        pt = (pt - pt.mean()) / pt.std()
        df = df.append(pd.DataFrame([pt],columns=column, index=[data.index[i+n]]), ignore_index=False)

    return df


def TrainDataSet(data, prior=1):
    x, y = [], []
    for i in range(len(data)-prior):
        a = data[i:(i+prior)]
        x.append(a)
        y.append(data[i + prior])
   
    trainX = np.array(x)
    trainY = np.array(y)    
    # RNN에 입력될 형식으로 변환한다. (데이터 개수, 1행 X prior 열)
    #trainX = np.reshape(trainX, (trainX.shape[0], 1, trainX.shape[1]))

    return trainX, trainY


def timestamp_gener(n,x) : #=time_premiumdata['timestamp'][-1]) : 
    time=[]
    for i in range(0,n) :
        time.append(x+360*i)
    return time