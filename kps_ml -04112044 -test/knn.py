# -*- coding: utf-8 -*-
import pandas as pd 
import matplotlib.pyplot as plt
import numpy as np
#from pandas.plotting import scatter_matrix
import datetime
import math
from scipy.stats import norm
from scipy import ndimage
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.utils import shuffle

data_for_knn=pd.read_csv("data/knndata.csv")
data_for_knn=shuffle(data_for_knn)
print(data_for_knn['premium'].groupby(data_for_knn['pre_class']).count())

x = data_for_knn.iloc[:, 3:17]
y = data_for_knn['pre_class']
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
    print('testAcc:',testAcc[-1]*100,'K:',k)
    # Train 세트의 Feature에 대한 정확도
    predY = knn.predict(trainX)
    trainAcc.append((trainY == predY).sum() / len(predY))
    print('trainAcc:',trainAcc[-1]*100,'K:',k)
#plt.figure(figsize=(8, 5))
#plt.plot(testAcc, label="Test Data")
#plt.plot(trainAcc, label="Train Data")
#plt.legend()
#plt.xlabel("k")
#plt.ylabel("Accuracy")
#plt.show()