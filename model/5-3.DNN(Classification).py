from keras import optimizers
from keras.models import Sequential
from keras.layers import Dense
from keras.utils import np_utils
import pandas as pd
import numpy as np
from sklearn.utils import shuffle
import matplotlib.pyplot as plt
from MyUtil import YahooData, FeatureSet 

# Yahoo site로부터 삼성전자 주가 데이터를 수집한다
# sam = YahooData.getStockData('005930.KS', '2007-01-01')
sam = YahooData.getStockDataYahoo ('^KS11', '2007-01-01')
# 저장된 파일을 읽어온다
#sam = pd.read_pickle('StockData/005930.KS')

# 주가 데이터 (OHLCV)로부터 기술적분석 지표들을 추출한다
# u = 0.8 : 수익률 표준편차의 0.8 배 이상이면 주가 상승 (class = 2)
# d = -0.8 : 수익률 표준편차의 -0.8배 이하이면 주가 하락 (class = 1)
# 아니면 주가 횡보 (classs = 0)
# ft = FeatureSet.getFeatureSet(sam, u=0.8, d=-0.7, period=20)
ft = FeatureSet.getFeatureSet(sam, u=0.6, d=-0.6, period=20)
# 분석할 데이터를 읽어와서 적당히 섞은 후 80%는 학습데이터로, 20%는 시험 데이터로 사용한다
ft = shuffle(ft)
nLen = len(ft)
n = int(nLen * 0.8) - 1
trainX = ft.iloc[0:n,0:6].values
trainY = np_utils.to_categorical(ft.iloc[0:n,6].values)
testX = ft.iloc[n:(nLen-1),0:6].values
testY = np_utils.to_categorical(ft.iloc[n:(nLen-1),6].values)

# 인공신경망 모델을 생성함.
model = Sequential()
model.add(Dense(20, input_dim=6, activation='relu'))
model.add(Dense(20, activation='relu'))
model.add(Dense(20, activation='relu'))
model.add(Dense(20, activation='relu'))
model.add(Dense(20, activation='relu'))
model.add(Dense(3, activation='sigmoid'))
adam = optimizers.Adam(lr = 0.005)
model.compile(loss='mse', optimizer=adam, metrics=['accuracy'])

# 학습 (Learning)
history = model.fit(trainX, trainY, batch_size = 50, epochs = 200, validation_data=(testX, testY))

# 학습 데이터 성능 곡선을 그린다
fig, ax1 = plt.subplots()
ax2 = ax1.twinx()
ax1.plot(history.history['acc'], color='red')
ax2.plot(history.history['loss'], color='blue')
ax1.set_xlabel("Epoch")
ax1.set_title("Performance of Training Data")
ax1.set_ylabel("Accuracy", color='red')
ax2.set_ylabel("Loss", color='blue')
plt.show()

# 시험 데이터 성능 곡선을 그린다
fig, ax1 = plt.subplots()
ax2 = ax1.twinx()
ax1.plot(history.history['val_acc'], color='red')
ax2.plot(history.history['val_loss'], color='blue')
ax1.set_xlabel("Epoch")
ax1.set_title("Performance of Test Data")
ax1.set_ylabel("Accuracy", color='red')
ax2.set_ylabel("Loss", color='blue')
plt.show()

# 금일 측정된 Feature가 아래와 같다면, 향후 주가의 방향은 ?
todayX = np.array([[-0.23,-1.45,0.85,0.43,-0.38,0.5]])
predY = model.predict(todayX)
predClass = np.argmax(predY)
print()
if predClass == 0:
    print("* 향후 주가는 횡보할 것으로 예상됨.")
elif predClass == 1:
    print("* 향후 주가는 하락할 것으로 예상됨.")
else:
    print("* 향후 주가는 상승할 것으로 예상됨.")

# 출력층의 출력값을 확률 척도로 변환함.
prob = predY / predY.sum()
np.set_printoptions(precision=4)
print("* 주가 횡보 확률 = %.2f %s" % (prob[0][0] * 100, '%'))
print("* 주가 하락 확률 = %.2f %s" % (prob[0][1] * 100, '%'))
print("* 주가 상승 확률 = %.2f %s" % (prob[0][2] * 100, '%'))

# weight 값을 출력해 본다
np.set_printoptions(precision=3)
for layer in model.layers:
    weights = layer.get_weights()
    print(weights)
    print()