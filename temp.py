# -*- coding: utf-8 -*-
import urllib.request as req
import json 
import pandas as pd
import time
from bs4 import BeautifulSoup
from function.mail_func import exchange, premiumFunc, mail


        
    
def crawling(x=bithumb,y=poloniex) :
    url_="https://api.bithumb.com/public/ticker/ALL"
    url_coinone="https://api.coinone.co.kr/ticker/?currency=all&format=json"
    url_poloniex="https://poloniex.com/public?command=returnTicker"
    url_exchange_rate="http://info.finance.naver.com/marketindex/?tabSel=exchange#tab_section"
    url=[url_bithumb, url_coinone, url_poloniex, url_exchange_rate]
    web_data={}
    err=0
    z=0
    e=0
        
    for i,j in enumerate(url) :
        res=req.urlopen(j)
        data=json.load(res)
        if i==0 :
            if data['status']=='0000' :
                web_data['bithumb']=data['data']
            else :
                err=10
            if data['errorCode']=='0' :
                web_data['coinone']=data
            else :
                err=11
        if i==2 : 
            #poloniex status 값
            web_data['poloniex'] =data
        if i==3 :
            exchange_rate=exchange()
            

    time.sleep(10)

    premium=premiumFunc(web_data['bithumb']['BTC']['sell_price'],web_data['poloniex']['USDT_BTC']['last'])
    print(
        "bithumb:", web_data['bithumb']['BTC']['sell_price'],
        "coinone:",web_data['coinone']['btc']['last'],
        "poloniex:",web_data['poloniex']['USDT_BTC']['last'],
        "premium:", premium)
    
    return premium, exchange_rate
#        if z==0 :
#            if premium > 0.03 :
#                mail('more than 3%')
#                z+=1
#        if e==0 : 
#            if premium < 0.02 :
#                mail('less than 2%')
#                e+=1
    


# coinone, kraken data 파일로 떨구는 함수 (한시간 마다 떨구면 될듯 )
def data_to_file() : 
    now_timestamp=str(time.time()).replace(".","")+'00'
    before_hour_timestamp=str(int(now_timestamp)-3600000000000)
    url_coinone = 'https://api.coinone.co.kr/trades/?currency=btc&period=hour&format=json'
    url_kraken = 'https://api.kraken.com/0/public/Trades?pair=XBTUSD&since='+before_hour_timestamp # since 값을 한시간씩 옮기면 될듯.
    
    
    res_coinone=req.urlopen(url_coinone)
    data_coinone=json.load(res_coinone)
    data_coinone=pd.DataFrame(data_coinone['completeOrders'])
    data_coinone=data_coinone[['timestamp','price','qty']]
    json_to_file(data_coinone,'coinone',1)
      
    
    res_kraken=req.urlopen(url_kraken)
    data_kraken=json.load(res_kraken)
    data_kraken=pd.DataFrame(data_kraken['result']['XXBTZUSD'])
    data_kraken=data_kraken[[2,0,1]]
    data_kraken.columns=['timestamp','price','qty']
    json_to_file(data_kraken,'kraken',1)


def exchange_rate_to_file(exchange_data,i):
    i=str(i)
    exchange_rate=pd.DataFrame(exchange_data)
    exchange_rate.to_csv('data/exchange_rate_'+i+'.csv') 
    
def json_to_file(data,coinone,i):
    i=str(i)
    data=pd.DataFrame(data)
    data.to_csv('data/'+coinone+i+'.csv') 

# while 문 돌면서 timestamp 랑 exchange rate crawling - 시간 간격 30분 ? if 문 안에 넣어서 떨구고 리셋도 함께하기 
time_exchange_data={'timestamp':[],'exchange_rate':[]}   
time_exchange_data['timestamp'].append(int(time.time()))
time_exchange_data['exchange_rate'].append(exchange())
