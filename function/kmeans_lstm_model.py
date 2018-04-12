# -*- coding: utf-8 -*-# -*- coding: utf-8 -*-
import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from keras.models import Sequential, load_model
from keras.layers import Dense, SimpleRNN
from keras.utils import to_categorical
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from keras.layers import LSTM
from function.ML import getClosePattern, realDataSet, TrainDataSet, timestamp_gener
from function.get_premium import get_premium
from function.insert_bitpred import insertpremium2, truncate

def Kmeans_LSTM(data) :
    

    data=pd.DataFrame(data)
    ft = getClosePattern(data, n=140)
    
    print("kmeans시작")

    k = 6
    km = KMeans(n_clusters=k, init='random', n_init=10, max_iter=500, tol=1e-04, random_state=0)
    km = km.fit(ft)
    y_km = km.predict(ft)
    ft['cluster'] = y_km
    
    
    


    print("1")
    nPrior =5
    data=ft['cluster'].values
    X, Y = TrainDataSet(data, nPrior)
    X=np.reshape(X,(len(X),nPrior,1))
    X=to_categorical(np.array(X))
    Y=to_categorical(np.array(Y))
    one_hot_vec_size=Y.shape[1]
    trainX, testX, trainY, testY = train_test_split(X, Y, test_size = 0.2, random_state=None)
    
    print("2")

    model = Sequential()
    model.add(LSTM(256, input_shape=(nPrior,k)))
    model.add(Dense(one_hot_vec_size, activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='adam',metrics=['accuracy'])
    history = model.fit(trainX, trainY, batch_size=1, epochs =10)
    
    
    print("3")
    
    predY = model.predict(testX)
    y_pr=[np.argmax(y, axis=None, out=None) for y in predY ]
    y_pr=np.array(y_pr)
    y_test=[np.argmax(y, axis=None, out=None) for y in testY ]
    y_test=np.array(y_test)
    accuracy = 100 * (y_test == y_pr).sum() / len(y_pr)
    print()
    print("* 시험용 데이터로 측정한 정확도 = %.2f" % accuracy, '%')
    
    
    
    test=get_premium(400)
    test=pd.DataFrame(test)
    test=test.sort_index(ascending=False)
    test_cluster=getClosePattern(test,n=140)
    y_t=km.predict(test_cluster) 
    
    print("4")


    a=realDataSet(y_t, nPrior)
    
    print("1")

    
    ttx=to_categorical(a)
    print("오류의심1")
    dY =model.predict(ttx)
    print("오류의심2")
    y_pr=[np.argmax(y, axis=None, out=None) for y in dY ]
    
    print("6")
    
    y_pr[-1]
    
    sigma=test['premium'][-140:].std() # 마지막 100개
    mu=list(test['premium'])[-1]-km.cluster_centers_[y_pr[-1],:][0]*sigma
    pred=km.cluster_centers_[y_pr[-1],:]*sigma+mu
    pred=pd.DataFrame(pred)
  
    print("7")
    
    last_timestamp=list(test['timestamp'])[-1]
    timestamp=list(map(int,timestamp_gener(140,last_timestamp)))
    premium=list(map(float,list(pred[0])))
    insertdata={'timestamp':timestamp,'premium':premium}
    
    print("8")
    truncate()
    print("9")
    insertpremium2(insertdata)
    print("10")
