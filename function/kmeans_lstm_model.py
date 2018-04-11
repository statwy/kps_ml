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
from function.insert_bitpred import insertpremium

def Kmeans_LSTM(data) :
    
#    data=pd.read_csv("data/data_refined.csv")
#    data.index=data['5min_0']
#    data=data.loc['2015-03-30 18:20:00' : ]
    data=pd.DataFrame(data)
    ft = getClosePattern(data, n=140)
    
    #Pattern 몇 개를 확인해 본다
    #x = np.arange(100)
    #plt.plot(x, ft.iloc[0])
    #plt.plot(x, ft.iloc[10])
    #plt.plot(x, ft.iloc[50])
    #plt.show()
    #print(ft.head())
    # K-means 알고리즘으로 Pattern 데이터를 8 그룹으로 분류한다 (k = 8)
    k = 6
    km = KMeans(n_clusters=k, init='random', n_init=10, max_iter=500, tol=1e-04, random_state=0)
    km = km.fit(ft)
    y_km = km.predict(ft)
    ft['cluster'] = y_km
    
    #ft.to_csv("data/kmeansdata.csv")
    
    
    # Centroid pattern을 그린다
   
    
    fig = plt.figure(figsize=(10, 6))
    
    for i in range(k):
        s = 'pattern-' + str(i)
        p = fig.add_subplot(2,3,i+1)
        p.plot(km.cluster_centers_[i,:], color="rbgkmrbgkm"[i])
        p.set_title('Cluster-' + str(i))
     
    plt.tight_layout()
    plt.show()
     
    

    
    nPrior =5
    data=ft['cluster'].values
    X, Y = TrainDataSet(data, nPrior)
    X=np.reshape(X,(len(X),nPrior,1))
    X=to_categorical(np.array(X))
    Y=to_categorical(np.array(Y))
    one_hot_vec_size=Y.shape[1]
    trainX, testX, trainY, testY = train_test_split(X, Y, test_size = 0.2, random_state=None)
    
    
    # RNN 모델 빌드 및 fitting
    model = Sequential()
    model.add(LSTM(256, input_shape=(nPrior,k)))
    model.add(Dense(one_hot_vec_size, activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='adam',metrics=['accuracy'])
    history = model.fit(trainX, trainY, batch_size=10, epochs =100)
    
    
    predY = model.predict(testX)
    y_pr=[np.argmax(y, axis=None, out=None) for y in predY ]
    y_pr=np.array(y_pr)
    y_test=[np.argmax(y, axis=None, out=None) for y in testY ]
    y_test=np.array(y_test)
    accuracy = 100 * (y_test == y_pr).sum() / len(y_pr)
    print()
    print("* 시험용 데이터로 측정한 정확도 = %.2f" % accuracy, '%')
    
    
#    model.save("data/model.h5")
#    a=model.get_weights("data/model.h5")
    
    
    #test=pd.read_csv("data/test.csv") # 
    
    test=get_premium(400)
    test=pd.DataFrame(test)
    test=test.sort_index(ascending=False)
    test_cluster=getClosePattern(test,n=140)
    y_t=km.predict(test_cluster) 
    
    #ansx,ansy = TrainDataSet(y_t, nPrior)
    #ttx=to_categorical(ansx)
    #dY =model.predict(ttx)
    #y_pr=[np.argmax(y, axis=None, out=None) for y in dY ]
    #y_pr=np.array(y_pr)
    
    a=realDataSet(y_t, nPrior)
    
    #a=list(a)
    #b=[]
    #b.append(a)
    
    
    ttx=to_categorical(a)
    dY =model.predict(ttx)
    y_pr=[np.argmax(y, axis=None, out=None) for y in dY ]
    
    y_pr[-1]
    
    sigma=test['premium'][-140:].std() # 마지막 100개
    mu=list(test['premium'])[-1]-km.cluster_centers_[y_pr[-1],:][0]*sigma
    pred=km.cluster_centers_[y_pr[-1],:]*sigma+mu
    pred=pd.DataFrame(pred)
    plt.plot(pred[0])
    
    
    
    last_timestamp=list(test['timestamp'])[-1]
    timestamp=list(map(int,timestamp_gener(100,last_timestamp)))
    premium=list(map(float,list(pred[0])))
    insertdata={'timestamp':timestamp,'premium':premium}
    
    insertpremium(insertdata)

