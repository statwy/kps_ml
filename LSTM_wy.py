# -*- coding: utf-8 -*-
from keras.models import Sequential
from keras.layers import Dense, LSTM
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

data=pd.read_csv("data/data_refined.csv")
data=data[['5min_0','premium']]

data['mpremium'] = data['premium'].rolling(20).mean()
data = data.dropna()

# Spread와 Spread의 5-기간 이동평균을 그린다
#plt.figure(figsize=(8, 3.5))
#plt.plot(data['premium'], color='blue', label='premium', linewidth=1)
#plt.plot(data['mpremium'], color='red', label='mspread(50)', linewidth=1)
#plt.legend()
#plt.show()

def TrainDataSet(data, prior=1):
    x, y = [], []
    for i in range(len(data)-prior):
        a = data[i:(i+prior)]
        x.append(a)
        y.append(data[i + prior])
    
    trainX = np.array(x)
    trainY = np.array(y)
    
    # RNN에 입력될 형식으로 변환한다. (데이터 개수, 1행 X prior 열)
    trainX = np.reshape(trainX, (trainX.shape[0], 1, trainX.shape[1]))

    return trainX, trainY

# 10개 데이터의 시퀀스로 다음 번 시계열을 예측함.
nPrior = 864

# 학습 데이터와 목표값 생성
data11= data['premium'].values
trainX, trainY = TrainDataSet(data11, nPrior)

# RNN 모델 빌드 및 fitting
model = Sequential()
model.add(LSTM(1000, input_shape=(1,nPrior)))
model.add(Dense(1))
model.compile(loss='mse', optimizer='adam')
history = model.fit(trainX, trainY, batch_size=200, epochs = 50)

# 향후 10 기간 데이터를 예측한다
nFuture = 100
dx = np.copy(data11)
estimate = [dx[-1]]
for i in range(nFuture):
    # 마지막 nPrior 만큼 입력데이로 다음 값을 예측한다
    x = dx[-nPrior:]
    x = np.reshape(x, (1, 1, nPrior))
    
    # 다음 값을 예측한다
    y = model.predict(x)[0][0]
    
    # 예측값을 저장해 둔다 
    estimate.append(y)
    
    # 이전 예측값을 포함하여 또 다음 값을 예측하기위해 예측한 값을 저장해 둔다
    dx = np.insert(dx, len(dx), y)

# 원 시계열의 마지막 부분 100개와 예측된 시계열을 그린다
print(estimate)


#dtail = data11[-100:]
#ax1 = np.arange(1, len(dtail) + 1)
#ax2 = np.arange(len(dtail), len(dtail) + len(estimate))
#plt.figure(figsize=(8, 7))
#plt.plot(ax1, dtail, color='blue', label='Spread', linewidth=1)
#plt.plot(ax2, estimate, color='red', label='Estimate')
#plt.axvline(x=ax1[-1],  linestyle='dashed', linewidth=1)
#plt.title('Spread & Estimate')
#plt.legend()
#plt.show()

