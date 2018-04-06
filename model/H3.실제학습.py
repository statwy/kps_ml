# 페어 트레이딩 대상 종목들의 실제 데이터로 학습한다.
# 가상 주가로 초기 학습된 신경망을 실제 데이터를 사용하여
# 추가로 학습 시킨다.
# -----------------------------------------------------------------
from keras.models import Sequential
from keras.layers import Dense, LSTM
import pandas as pd
import numpy as np

pair = pd.read_csv("P.Pair종목(200)_hy.csv", engine='python')

def npiSpread(codeA, codeB):
    # 해당 종목의 주가 데이터를 가져온다
    p1 = pd.read_pickle('StockData/' + codeA)
    p2 = pd.read_pickle('StockData/' + codeB)
    
    # 해당 종목의 NPI 주가, NPI 스프레드를 계산한다
    s = pd.DataFrame(columns=['npiA', 'npiB', 'spread'])
    s['npiA'] = (p1['Close'] - p1['Close'].mean()) / p1['Close'].std()
    s['npiB'] = (p2['Close'] - p2['Close'].mean()) / p2['Close'].std()
    s['spread'] = s['npiA'] - s['npiB']
    s = s.dropna()
    return s

# RNN 입력값과 출력 목표값을 생성한다
# ex : s = ([1,2,3,4,5,6,7,8,9,10])
#      x, y = TrainDataSet(s, prior = 3)
#
# x = array([[[1, 2, 3]],    y = array([4,   --> 1,2,3 다음에는 4가 온다
#            [[2, 3, 4]],               5,   --> 2,3,4 다음에는 5가 온다
#            [[3, 4, 5]],               6,
#            [[4, 5, 6]],               7,
#            [[5, 6, 7]],               8,
#            [[6, 7, 8]],               9,
#            [[7, 8, 9]]])              10,])
#
# 3개 데이터의 시퀀스를 이용하여 다음 시계열을 예측하기 위한 예시임.
# ----------------------------------------------------------------------
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

# LSTM 모델을 빌드한다
def buildModel(nInput):
    model = Sequential()
    model.add(LSTM(10, input_shape=(1,nInput)))
    model.add(Dense(1))
    model.compile(loss='mse', optimizer='adam')
    return model

def simLearning(data):
    for x in data.itertuples():
        nPrior = 10     # 과거 10-기간 데이터로 미래 예측
        
        # LSTM 모델을 빌드한다
        model = buildModel(nPrior)
        
        # LSTM weight의 초깃값을 결정한다. 저장된 weight가 있으면 이를 적용한다
        weightFile = x.codeA[0:6] + '-' + x.codeB[0:6] + '.h5'
        try:
            model.load_weights("data/" + weightFile)
            print("기존 학습 결과 Weight를 적용하였습니다.")
        except:
            print("model Weight을 Random 초기화 하였습니다.")
                
        # 실제 주가의 NPI 스프레드를 생성하여, 초기 학습한다.
        df = npiSpread(x.codeA, x.codeB)
        
        # 학습 데이터를 구성한다
        trainX, trainY = TrainDataSet(df['spread'].values, prior = nPrior)
         
        # LSTM 모델을 학습한다
        history = model.fit(trainX, trainY, batch_size=100, epochs = 300, verbose=False)
            
        # 학습 결과를 저장해 둔다
        model.save_weights("data/" + weightFile)
        
        print("%s - %s 실제 학습 완료." % (x.stockA, x.stockB))
            