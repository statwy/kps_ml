

# -*- coding: utf-8 -*-
"""
Created on Fri Apr  6 11:29:00 2018

@author: user
"""

from keras.models import Sequential
from keras import optimizers
from keras.layers import Dense, LSTM
import pandas as pd
import numpy as np
import itertools as it
from keras.utils import np_utils
from sklearn.utils import shuffle
import matplotlib.pyplot as plt

#data=pd.read_csv("data/final_featureset_input_model.csv")
#trainX=pd.read_csv("data/trainX.csv")
#trainY=pd.read_csv("data/trainY.csv")
# LSTM 모델을 빌드한다
def buildModel(nInput):
    model = Sequential()
    #model.add(LSTM(6, input_shape=(8,nInput)))
    model.add(LSTM(128, input_shape=(nInput, 8))) # (timestep, feature)
    model.add(Dense(1))
    model.compile(loss='mse', optimizer='adam')
    return model

#def TrainDataSet(data, prior=1):
#    x, y = [], []
#    for i in range(len(data)-prior):
#        a = data[i:(i+prior)]
#        x.append(a)
#        y.append(data[i + prior])
#    
#    trainX = np.array(x)
#    trainY = np.array(y)
#    
#    # RNN에 입력될 형식으로 변환한다. (데이터 개수, 1행 X prior 열)
#    trainX = np.reshape(trainX, (trainX.shape[0], 1, trainX.shape[1]))
#
#    return trainX, trainY
    
def TrainDataSet(data, prior):
    ft = shuffle(data)
    #ft = ft.iloc[0:16000,:]
    nLen = len(ft)
    n = int(nLen * 0.8) - 1
    trainX = ft.iloc[0:n,0:8].values
    #trainX= trainX[0:16000]
    print(trainX.shape[0])
    trainY = ft.iloc[0:n,8].values
    #trainY= trainX[0:16000]
    #testX = ft.iloc[n:(nLen-1),0:8].values
    #testY = ft.iloc[n:(nLen-1),8].values
    #trainX = np.reshape(trainX, (trainX.shape[0], 1, trainX.shape[1]))
    trainX = np.array(trainX)
    trainX = np.reshape(trainX,(-1, prior, trainX.shape[1]))
    #trainX = np.reshape(trainX, (trainX.shape[0], 1, trainX.shape[1]))    #(size, timestamp, feature)
    #testX=np.reshape(testX, (testX.shape[0], prior, 8))    
    return trainX, trainY



def simLearning(data):
    for x in data.itertuples():
        nPrior = 10   # 과거 10-기간 데이터로 미래 예측
        
        # LSTM 모델을 빌드한다
        model = buildModel(nPrior)
        
        # LSTM weight의 초깃값을 결정한다. 저장된 weight가 있으면 이를 적용한다
        weightFile = 'w_sample01'+ '.h5'
        try:
            model.load_weights("data/" + weightFile)
            print("기존 학습 결과 Weight를 적용하였습니다.")
        except:
            print("model Weight을 Random 초기화 하였습니다.")
                
        # 실제 주가의 NPI 스프레드를 생성하여, 초기 학습한다.
#        df = npiSpread(x.codeA, x.codeB)
        
        #학습 데이터를 구성한다
        trainX, trainY= TrainDataSet(data, nPrior)
         
        # LSTM 모델을 학습한다
        model.fit(trainX, trainY, batch_size=100, epochs = 300, verbose=False)
            
        # 학습 결과를 저장해 둔다
        model.save_weights("data/" + weightFile)
        
        print("실제 학습 완료.")

data=pd.read_csv("data/inputdataset_final_9.csv")
simLearning(data)