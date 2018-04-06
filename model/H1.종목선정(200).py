# Yahoo site로 부터 대형주 종목 데이터를 수집하여 
# 페어 트레이딩 대상 종목을 찾는다.
#
# 한국생산성본부 금융 빅데이터 전문가 과정 (금융 모델링 파트) 실습용 코드
# Written : 2018.2.5
# 제작 : 조성현
# -----------------------------------------------------------------
import pandas as pd
import numpy as np
import itertools as it
from MyUtil.YahooData import getStockDataYahoo

df = pd.read_csv("kospi200.csv")
stocks = df.set_index('code').T.to_dict('list')

# Yahoo site로부터 대형주 종목 데이터를 수집한다
# 수정주가를 반영하여 삼성전자 주가 데이터를 수집한다. 가끔 안들어올 때가 있어서 10번 시도한다.
# Yahoo site로부터 대형주 종목 데이터를 수집한다
# 수정주가를 반영하여 삼성전자 주가 데이터를 수집한다. 가끔 안들어올 때가 있어서 10번 시도한다.
def collectData(startDate="2007-01-01"):
    n = 1
    for key in stocks.keys():
        code = str(key).rjust(6, '0')
        
        # Yahoo site로부터 대형주 종목 데이터를 수집한다
        s = getStockDataYahoo(code + '.KS', start=startDate)
        print("%d) %s (%s.KS)가 수집되었습니다." % (n, stocks[key][0], code))
    
        # 수집한 데이터를 파일에 저장한다.
        s.to_pickle('StockData/' + code + '.KS')
        n += 1

# 저장된 주가 데이터를 읽어와서 종목 별로 연평균 수익률, 변동성 Sharp ratio를 계산한다
def findPair():
    result = pd.DataFrame(columns=['codeA', 'codeB', 'stockA', 'stockB', 'prcCor', 'rtnCor', 'volCor', 'sum'])
    
    n = 0
    for pair in it.combinations(stocks.keys(), 2):
        # 저장된 주가 데이터를 읽어온다
        p1 = str(pair[0]).rjust(6, '0') + '.KS'
        s1 = pd.read_pickle('StockData/' + p1)
        p2 = str(pair[1]).rjust(6, '0') + '.KS'
        s2 = pd.read_pickle('StockData/' + p2)

        if s1.shape[0] < 500 or s2.shape[0] < 500:
            continue
            
        # Pair 데이터 프레임을 구성한다
        pdf = pd.DataFrame(columns=['stockA', 'stockB', 'volA', 'volB'])
        pdf['stockA'] = s1['Close']
        pdf['stockB'] = s2['Close']
        pdf['volA'] = s1['Volume']
        pdf['volB'] = s2['Volume']
        pdf = pdf.dropna()
        
        # 수익률 상관계수를 측정한다
        pdf['rtnA'] = np.log(pdf['stockA']) - np.log(pdf['stockA'].shift(1))
        pdf['rtnB'] = np.log(pdf['stockB']) - np.log(pdf['stockB'].shift(1))
        pdf = pdf.dropna()
        rtnCor = np.corrcoef(pdf['rtnA'], pdf['rtnB'])[1,0]

        # 로그 가격의 상관계수를 측정한다
        prcCor = np.corrcoef(np.log(pdf['stockA']), np.log(pdf['stockB']))[1,0]
        
        # 거래량 상관계수를 측정한다
        volCor = np.corrcoef(pdf['volA'], pdf['volB'])[1,0]
        
        # 결과를 종합한다
        sumCor = prcCor + rtnCor + volCor
        result.loc[n] = [p1, p2, stocks[pair[0]], stocks[pair[1]], prcCor, rtnCor, volCor, sumCor]
        n += 1
        
        print("%d) (%s - %s)가 처리되었습니다." % (n, pair[0], pair[1]))
        
    return result

p = findPair()
s = p.sort_values(['sum'], ascending=False).head(50)
s.to_csv('P.Pair종목(200)_hy.csv')
