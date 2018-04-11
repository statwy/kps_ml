# -*- coding: utf-8 -*-
import pandas as pd 
import numpy as np
#import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from keras.models import Sequential
from keras.layers import Dense, SimpleRNN
from keras.utils import to_categorical
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from keras.layers import LSTM

print("kmeans 시작")
#
def getClosePattern(data, n):
    loc = tuple(range(1, len(data) - n, 20))    
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



data=pd.read_csv("data/data_refined.csv")
data.index=data['5min_0']
data=data.loc['2015-03-30 18:20:00' :]
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
print("kmeans 완료")

#plt.plot(list(y_km))

#########################################

#test=pd.read_csv("data/test.csv")
#test_cluster=getClosePattern(test,n=120)
#y_t=km.predict(test_cluster)



#########################################

# Centroid pattern을 그린다

fig = plt.figure(figsize=(10, 6))

for i in range(k):
    s = 'pattern-' + str(i)
    p = fig.add_subplot(2,4,i+1)
    p.plot(km.cluster_centers_[i,:], color="rbgkmrbgkm"[i])
    p.set_title('Cluster-' + str(i))
 
plt.tight_layout()
plt.show()
 
# cluster = 0 인 패턴 몇 개만 그려본다
#cluster = 0
#plt.figure(figsize=(8, 5))
#p = ft.loc[ft['cluster'] == cluster]
#for i in range(7):
#    plt.plot(x,p.iloc[i][0:100])
#   
#plt.title('Cluster-' + str(cluster))
#plt.show()
#
#plt.hist(ft['cluster'])
#plt.show()




##################################################################################################
##################################################################################################

# Train 데이터 세트와 Test 데이터 세트를 구성한다

#################################################################################################


#data=pd.read_csv("data/data_refined.csv")
#data=data[['5min_0','premium']]

#data['mpremium'] = data['premium'].rolling(20).mean()
#data = data.dropna()

# Spread와 Spread의 5-기간 이동평균을 그린다
#plt.figure(figsize=(8, 3.5))
#plt.plot(data['premium'], color='blue', label='premium', linewidth=1)
#plt.plot(data['mpremium'], color='red', label='mspread(50)', linewidth=1)
#plt.legend()
#plt.show()

#def TrainDataSet(data, prior=1):
#    x, y = [], []
#    for i in range(len(data)-prior):
#        a = data[i:(i+prior)]
#        x.append(a)
#        y.append(data[i + prior])
#   
#    trainX = np.array(x)
#    trainY = np.array(y)    
#    # RNN에 입력될 형식으로 변환한다. (데이터 개수, 1행 X prior 열)
#    #trainX = np.reshape(trainX, (trainX.shape[0], 1, trainX.shape[1]))
#
#    return trainX, trainY


nPrior =10
data=ft['cluster'].values

X, Y = TrainDataSet(data, nPrior)
X=np.reshape(X,(len(X),nPrior,1))
X=to_categorical(np.array(X))
Y=to_categorical(np.array(Y))

#X.shape
one_hot_vec_size=Y.shape[1]
trainX, testX, trainY, testY = train_test_split(X, Y, test_size = 0.2, random_state=None)
print("훈련시작")
# RNN 모델 빌드 및 fitting
model = Sequential()
#model.add(LSTM(128, input_shape=(nPrior,k)))
model.add(LSTM(512, input_shape=(nPrior,k)))
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


#predY = model.predict(trainX)
#y_pr=[np.argmax(y, axis=None, out=None) for y in predY ]
#y_pr=np.array(y_pr)
#y_train=[np.argmax(y, axis=None, out=None) for y in trainY ]
#y_train=np.array(y_test)
#accuracy = 100 * (y_train == y_pr).sum() / len(y_pr)
#print("* 학습용 데이터로 측정한 정확도 = %.2f" % accuracy, '%')





## RNN 모델 빌드 및 fitting
#model = Sequential()
#model.add(Dense(20, input_shape=(1,nPrior)))
#model.add(Dense(1))
#model.compile(loss='categorical_crossentropy', optimizer='rmsprop',metrics=['accuracy'])
#history = model.fit(trainX, trainY, batch_size=1, epochs =100)
#
#predY = model.predict(testX)
#pred=[]
#for i in range(0,len(predY)) :
#    pred.append(int(round(predY[i][0])))
#accuracy = 100 * (testY == pred).sum() / len(predY)
#print()
#print("* 시험용 데이터로 측정한 정확도 = %.2f" % accuracy, '%')
#
## Train 세트의 Feature에 대한 class를 추정하고, 정확도를 계산한다
#predY = knn.predict(trainX)
#accuracy = 100 * (trainY == predY).sum() / len(predY)
#print("* 학습용 데이터로 측정한 정확도 = %.2f" % accuracy, '%')


# 향후 10 기간 데이터를 예측한다
#nFuture = 1
#dx = np.copy(data)
#estimate = [dx[-1]]
#for i in range(nFuture):
#    # 마지막 nPrior 만큼 입력데이로 다음 값을 예측한다
#    x = dx[-nPrior:]
#    x = np.reshape(x, (1, 1, nPrior))
#    
#    # 다음 값을 예측한다
#    y = model.predict(x)[0][0]
#    
#    # 예측값을 저장해 둔다 
#    estimate.append(y)
#    
#    # 이전 예측값을 포함하여 또 다음 값을 예측하기위해 예측한 값을 저장해 둔다
#    dx = np.insert(dx, len(dx), y)
#
## 원 시계열의 마지막 부분 100개와 예측된 시계열을 그린다
#print(estimate)
#
#
#dtail = data[-100:]
#ax1 = np.arange(1, len(dtail) + 1)
#ax2 = np.arange(len(dtail), len(dtail) + len(estimate))
#plt.figure(figsize=(8, 7))
#plt.plot(ax1, dtail, color='blue', label='Spread', linewidth=1)
#plt.plot(ax2, estimate, color='red', label='Estimate')
#plt.axvline(x=ax1[-1],  linestyle='dashed', linewidth=1)
#plt.title('Spread & Estimate')
#plt.legend()
#plt.show()