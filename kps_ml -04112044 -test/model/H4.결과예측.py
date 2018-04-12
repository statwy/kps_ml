# 페어 트레이딩 대상 종목들의 실제 데이터로 학습한다.
# 가상 주가로 초기 학습된 신경망을 실제 데이터를 사용하여
# 추가로 학습 시킨다.
# -----------------------------------------------------------------
from keras.models import Sequential
from keras.layers import Dense, LSTM
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# plot에서 한글 처리를 위해 아래 폰트를 사용한다
from matplotlib import font_manager, rc
font_name = font_manager.FontProperties(fname="c:/Windows/Fonts/malgun.ttf").get_name()
rc('font', family=font_name)

pair = pd.read_csv("P.Pair종목(200).csv", engine='python')

def npiSpread(codeA, codeB):
    # 해당 종목의 주가 데이터를 가져온다
    p1 = pd.read_pickle('StockData/' + codeA)
    p2 = pd.read_pickle('StockData/' + codeB)
    
    # 해당 종목의 NPI 주가, NPI 스프레드를 계산한다
    s = pd.DataFrame(columns=['npiA', 'npiB', 'spread'])
    s['npiA'] = (p1['Close'] - p1['Close'].mean()) / p1['Close'].std()
    s['npiB'] = (p2['Close'] - p2['Close'].mean()) / p2['Close'].std()
    s['spread'] = s['npiA'] - s['npiB']
    
    return s

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
                
        # 실제 주가의 NPI 스프레드를 생성하고 마지막 nPrior개 데이터를
        # 신경망에 입력하여 nFuture 기간의 스프레드를 예측한다
        nFuture = 10
        df = npiSpread(x.codeA, x.codeB)
        df = df.dropna()
        df = df['spread'].values
        
        dx = np.copy(df)
        estimate = [dx[-1]]
        for i in range(nFuture):
            # 마지막 nPrior 만큼 입력데이로 다음 값을 예측한다
            xInput = dx[-nPrior:]
            xInput = np.reshape(xInput, (1, 1, nPrior))
            
            # 다음 값을 예측한다.
            y = model.predict(xInput)[0][0]
            print(xInput, y)
            # 예측값을 저장해 둔다
            estimate.append(y)
    
            # 이전 예측값을 포함하여 또 다음 값을 예측하기위해 예측한 값을 저장해 둔다
            dx = np.insert(dx, len(dx), y)
            
        # 원 시계열의 마지막 부분 100개와 예측된 시계열을 그린다
        dtail = df[-100:]
        ax1 = np.arange(1, len(dtail) + 1)
        ax2 = np.arange(len(dtail), len(dtail) + len(estimate))
        plt.figure(figsize=(8, 7))
        plt.plot(ax1, dtail, color='blue', label='Spread', linewidth=1)
        plt.plot(ax2, estimate, color='red', label='Estimate')
        plt.axvline(x=ax1[-1],  linestyle='dashed', linewidth=1)
        plt.title(x.stockA + '-' + x.stockB)
        plt.legend()
        plt.show()
        #print("추정치 : ", estimate)
        
# weight 값을 출력해 본다
def checkWeight(codeA, codeB):
    # LSTM 모델을 빌드한다
    model = buildModel(10)
    
    # LSTM weight의 초깃값을 결정한다. 저장된 weight가 있으면 이를 적용한다
    weightFile = codeA + '-' + codeB + '.h5'
    try:
        model.load_weights("data/" + weightFile)
        print("기존 학습 결과 Weight를 적용하였습니다.")
    except:
        print("model Weight을 Random 초기화 하였습니다.")
            
    np.set_printoptions(precision=3)
    for layer in model.layers:
        weights = layer.get_weights()
        print(weights)
        print()