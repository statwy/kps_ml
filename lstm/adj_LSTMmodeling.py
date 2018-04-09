# -*- coding: utf-8 -*-

from keras.models import Sequential
from keras import optimizers
from keras.layers import Dense, LSTM, Dropout
import pandas as pd
import numpy as np
import itertools as it
from keras.utils import np_utils
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt

df = pd.read_csv("inputdataset_final.csv")
print( df.shape ) # (164009, 11)
# df.head()

df = df[8:] # 짝수로 숫자 맞추기 위해 앞의 8개 빼기


df['premium'].isnull().sum() # 1개의 결측치가 있음!

df = df[pd.notnull(df['premium'])] # 1개의 결측치 삭제

# pandas.core.series.Series > numpy.ndarray > list
premium = df['premium'].values.tolist()

print(type(premium)) # <class 'list'>
print(len(premium)) # 164000


window_size = 500
X = []
Y = []
print('len(premium) - window_size=',  len(premium) - window_size ) # len(premium) - window_size= 163995
for i in range(len(premium) - window_size):
    # i = 0
    # j = 0, 1, 2, 3, 4
    X.append([premium[i+j] for j in range(window_size)])
    Y.append(premium[window_size+i])
    
    
X = np.array(X)
Y = np.array(Y)

print( X.shape ) # (163995, 5)
print( Y.shape ) # (163995,)

print( X )
print( Y )



train_test_split = 120000

X_train = X[:train_test_split,:]
Y_train = Y[:train_test_split]

X_test = X[train_test_split:,:]
Y_test = Y[train_test_split:]

print( "X_train.shape=", X_train.shape ) # X_train.shape= (160000, 5)
print( "X_test.shape=", X_test.shape )   # X_test.shape= (3995, 5)
print( "Y_train.shape=", Y_train.shape ) # Y_train.shape= (160000,)
print( "Y_test.shape=", Y_test.shape )   # Y_test.shape= (3995,)

print( X_train[0] )
print( X_test[0] )
print( Y_train[0] )
print( Y_test[0] )




X_train = np.reshape(X_train, (X_train.shape[0], window_size, 1))
X_test  = np.reshape(X_test, (X_test.shape[0], window_size, 1))

print( "X_train.shape=", X_train.shape ) # X_train.shape= (160000, 5, 1)
print( "X_test.shape=", X_test.shape )   # X_test.shape= (3995, 5, 1)
print( "Y_train.shape=", Y_train.shape ) # Y_train.shape= (160000,)
print( "Y_test.shape=", Y_test.shape )   # Y_test.shape= (3995,)
print( X_train[0] )
print( X_test[0] )



model = Sequential()
model.add(LSTM(128, input_shape=(5,1,)))
model.add(Dropout(0.2))
model.add(Dense(1, activation='linear'))
model.compile(loss='mse', optimizer='adam')
model.summary()



import time
start_time = time.time() 

hist = model.fit(X_train, Y_train, epochs=1, validation_split=0.1)
# hist = model.fit(X_train, Y_train, epochs=10, validation_split=0.1)
# hist = model.fit(X_train, Y_train, epochs=6, batch_size=1, validation_split=0.1)
print(hist.history)

lap = (time.time() - start_time)
m, s = divmod(lap, 60)
h, m = divmod(m, 60)
print( "%d:%02d:%02d" % (h, m, s) )  # 예) 1번=0:00:59, 10번=0:09:38


score = model.evaluate(X_train, Y_train)
score


score2 = model.evaluate(X_test, Y_test)
score2


train_predict = model.predict(X_train)


y_pred = model.predict(X_test)



print(X_test.shape) # (3995, 5, 1)
print(Y_test.shape) # (3995,)
print(Y_test[0]) # 4.2752522040000001
print(y_pred[0]) # array([ 4.23595238], dtype=float32)



mean_squared_error(Y_test, y_pred)



Y_test


y_pred


plt.figure(figsize=(15,10))
plt.plot(premium)

split_pt = train_test_split+window_size
plt.plot(np.arange(window_size, split_pt, 1), train_predict, color='g')

plt.plot(np.arange(split_pt, split_pt+len(y_pred), 1), y_pred, color='r')