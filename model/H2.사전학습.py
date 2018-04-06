# 페어 트레이딩 대상 종목들의 특성을 이용하여 사전학습한다
# 해당 페어와 동등한 상관계수를 갖는 두 주가를 생성하여 신경망을
# 초기학습한다.
# -----------------------------------------------------------------
from keras.models import Sequential
from keras.layers import Dense, LSTM
import pandas as pd
import numpy as np
import itertools as it

#pair = pd.read_csv("P.Pair종목(200).csv", engine='python')
pair = pd.read_csv("P.Pair종목(200)_hy", engine='python')

# 초기값 S0 에서 다음 GBM값 1개를 계산한다. drift, vol은 연간 단위
def GBM(w, drift, vol, S0=1):
    mu = drift / 252             # daily drift rate
    sigma = vol / np.sqrt(252) 	# daily volatlity 
    
    # Monte Carlo simulation
    S = S0 * np.exp((mu - 0.5 * sigma**2) + sigma * w)
    return S

# n-기간 동안의 가상 주가 2개를 생성한다
def CorStockPrice(n, corr, r1, r2, s1, s2, S0):
    sA = []
    sB = []
    S0A = S0
    S0B = S0
    for i in range(0, n):
        wA = np.random.normal(0, 1, 1)[0]
        wB = np.random.normal(0, 1, 1)[0]
        zA = wA
        zB = wA * corr + wB * np.sqrt(1 - corr ** 2)
        
        pA = GBM(zA, r1, s1, S0A)
        pB = GBM(zB, r2, s2, S0B)
        
        sA.append(pA)
        sB.append(pB)
        
        S0A = pA
        S0B = pB
        
    # 가상 주가를 기록해 둔다
    s = pd.DataFrame(sA, columns = ['A'])
    s['B'] = sB
    
    # 가상 주가로 NPI 주가, NPI 스프레드를 계산한다
    s['npiA'] = (s['A'] - s['A'].mean()) / s['A'].std()
    s['npiB'] = (s['B'] - s['B'].mean()) / s['B'].std()
    s['spread'] = s['npiA'] - s['npiB']
    
    return s

# RNN 입력값과 출력 목표값을 생성한다
# ex : s = ([1,2,3,4,5,6,7,8,9,10])
#      x, y = TrainDataSet(s, prior = 3)
#
# x = array([[[1, 2, 3]],    y = array([4,   --> 1,2,3 다음에는 4가 온다
#            [[2, 3, 4]],               5,   --> 2,3,4 다음에는 5가 온다
#          .000000000000000000000000000000000000000000000000000000000000000000009/.++++  [[3, 4, 5]],               6,
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

# 지정된 페어의 수익률, 변동성을 측정한다
def findRtnVol(codeA, codeB):
    # 해당 종목의 주가 데이터를 가져온다
    p1 = pd.read_pickle('StockData/' + codeA)
    p2 = pd.read_pickle('StockData/' + codeB)

    # 일일 수익률을 계산한다
    p1['Rtn'] = np.log(p1['Close']) - np.log(p1['Close'].shift(1))
    p2['Rtn'] = np.log(p2['Close']) - np.log(p2['Close'].shift(1))
    p1 = p1.dropna()
    p2 = p2.dropna()
    
    # 연 평균 수익률을 계산한다
    r1 = p1['Rtn'].mean() * 252
    r2 = p2['Rtn'].mean() * 252
    
    # 변동성을 계산한다
    s1 = p1['Rtn'].std() * np.sqrt(252)
    s2 = p2['Rtn'].std() * np.sqrt(252)
    
    return r1, r2, s1, s2

# LSTM 모델을 빌드한다
def buildModel(nInput):
    model = Sequential()
    model.add(LSTM(10, input_shape=(1,nInput)))
    model.add(Dense(1))
    model.compile(loss='mse', optimizer='adam')
    return model

def simLearning(data):
    for x in data.itertuples():
        # 종목-1, 종목-2 코드로 수익률, 변동성, 상관계수를 측정한다
        corr = x.rtnCor # 두 중목의 수익률의 상관계수
        r1, r2, s1, s2 = findRtnVol(x.codeA, x.codeB) #findRtnVOl : codeA와 codeB의 주가를 수익률로 바꿔서 평균 수익률, 표준 편차를 계산한다
        
        nPrior = 10     # 과거 10-기간 데이터로 미래 예측
        
        # LSTM 모델을 빌드한다
        model = buildModel(nPrior)
        
        # LSTM weight의 초깃값을 결정한다. 저장된 weight가 있으면 이를 적용한다
        weightFile = x.codeA[0:6] + '-' + x.codeB[0:6] + '.h5' #h5 포멧으로 저장한다 
        try:
            model.load_weights("data/" + weightFile)
            print("기존 학습 결과 Weight를 적용하였습니다.")
        except:
            print("model Weight을 Random 초기화 하였습니다.")
                
        # 가상 주가, 가상 NPI 스프레드를 생성하여, 초기 학습한다.
        df = CorStockPrice(500, corr, r1, r2, s1, s2, 1000)
        
        # 학습 데이터를 구성한다
        trainX, trainY = TrainDataSet(df['spread'].values, prior = nPrior)
         
        # LSTM 모델을 학습한다
        history = model.fit(trainX, trainY, batch_size=100, epochs = 300, verbose=False)
            
        # 학습 결과를 저장해 둔다
        model.save_weights("data/" + weightFile)
        
        print("%s - %s 사전 학습 완료." % (x.stockA, x.stockB))
        
def preLearn(n):
    for i in range(n):
        simLearning(pair)
    