# -*- coding: utf-8 -*-
import pandas as pd
import matplotlib.pyplot as plt
from function.crawling_for_mail  import crawling
from function.crawling_for_mail  import data_to_file, exchange_rate_to_file
from function.mail_func import mail, exchange
from function.get_memberlist import get_memberlist
import math
import datetime
import time
from function.insert import insertpremium 
from function.main_func import bollingerband, append_maxsize, premium_int

def gener_percentFlag(x,y) :
    global percent_flag
    percent_flag={}
    percent_flag['Boll']=1
    for i in range(x,y+1) :
       a='p_%d' %i
       percent_flag[a]=1
       
def setup_percentFlag(x,y,z) :
    for i in range(x,y+1) :
        a='p_%d' %i
        percent_flag[a]=1
    b='p_%d' %z
    percent_flag[b]=0
    percent_flag['Boll']=0

# 초기 데이터 생성용 로직

j=0
premium=[]
for j in range(0,100) : 
    time.sleep(10)
    premium.append(crawling('bithumb','bitfinex'))
    j+=1
upper_bound,lower_bound=bollingerband(premium,100,4)

print('##### 메일 보내는 로직 시작 #####')
i=0
j=0
m=0
k=0
gener_percentFlag(-10,10)
time_exchange_data={'timestamp':[],'exchange_rate':[]} 
time_premiumdata={'timestamp':[],'premium':[]}
start_time_m=datetime.datetime.now()
start_time_h=datetime.datetime.now()
start_time_min=datetime.datetime.now()
while True :
    try :
        time.sleep(10)
        i+=1    
        premium=append_maxsize(premium,crawling('bithumb','bitfinex'),5000)
        upper_bound,lower_bound=bollingerband(premium,100,4)
        
        if not lower_bound[-1] < crawling('bithumb','bitfinex') < upper_bound[-1]:
            percent_flag['Boll']=1
                    
        if not math.floor(premium[-2]*100)==math.floor(premium[-1]*100) :
            temp=premium_int(premium[-2],premium[-1])
            temp_premium='p_%d' %temp
            print("temp 값 :", temp)
            if percent_flag['Boll']==1 or percent_flag[temp_premium]==1  :
                mail_address=get_memberlist(temp)
                #mail_address=['jwy627@naver.com']
                mail_content='현재 코리아 프리미엄'+'%d'%temp +'퍼센트 입니다'  
                mail(mail_content,'kps 알람입니다. 현재 프리미엄 %d 프로!'%temp , mail_address)          
                setup_percentFlag(-10,10,temp)
        
        #print("percent_flag 값:",percent_flag)
        time_for_logic_min=datetime.datetime.now()-start_time_min
        if time_for_logic_min.total_seconds()>300 :
            time_premiumdata['timestamp'].append(int(time.time()))
            print(int(time.time()))
            time_premiumdata['premium'].append(premium[-1]*100)
            insertpremium(time_premiumdata) # DB 에 프리미엄 넣기
            start_time_min=datetime.datetime.now()
            time_premiumdata={'timestamp':[],'premium':[]}
        time_for_logic_m=datetime.datetime.now()-start_time_m
        if time_for_logic_m.total_seconds()>3600 :   
            time_exchange_data['timestamp'].append(int(time.time()))
            time_exchange_data['exchange_rate'].append(exchange())
            #df_time_premiumdata=pd.DataFrame(time_premiumdata)
            #df_time_premiumdata.to_csv('data/premium'+str(m)+'.csv')    
            try :         
                data_to_file('coinone',m)         
            except :
                time.sleep(10)
                data_to_file('coinone',m)
                
            try :
                data_to_file('kraken',m)         
            except :
                time.sleep(10)
                data_to_file('kraken',m)        
            
            m+=1
            #time_premiumdata={'timestamp':[],'premium':[]}
            start_time_m=datetime.datetime.now()      
        time_for_logic_h=datetime.datetime.now()-start_time_h
        if time_for_logic_h.total_seconds()>3600*12 :
            exchange_rate_to_file(time_exchange_data,j)
            time_exchange_data={'timestamp':[],'exchange_rate':[]} 
            start_time_h=datetime.datetime.now()
            j+=1
    except :
        print("전체 로직 에러 ")
#    if i==2000 :
#        break


# bollinger band test plot
plt.plot(upper_bound)
plt.plot(premium)
plt.plot(lower_bound)    


#함수, 나중에 다른 폴더로 뺄 계획 
##############################################################################

#def bollingerband(x,y=100,sigma=5) :
#    x=pd.DataFrame(x)
#    mva=x.rolling(window=y).mean()
#    mvstd=x.rolling(window=y).std()
#    upper_bound=mva+sigma*mvstd
#    lower_bound=mva-sigma*mvstd
#    return list(upper_bound[0]), list(lower_bound[0])
#
#
#def append_maxsize(x,y,z) :
#    if len(x)<z :
#        x.append(y)
#    else :
#        x.pop(0)
#        x.append(y) 
#    return x
#
#
#def gener_percentFlag(x,y) :
#    global percent_flag
#    percent_flag={}
#    percent_flag['Boll']=1
#    for i in range(x,y+1) :
#       a='p_%d' %i
#       percent_flag[a]=1
#
#def setup_percentFlag(x,y,z) :
#    for i in range(x,y+1) :
#        a='p_%d' %i
#        percent_flag[a]=1
#    b='p_%d' %z
#    percent_flag[b]=0
#    percent_flag['Boll']=0
#
#       
#def premium_int(x,y) : 
#    z= max(math.floor(x*100),math.floor(y*100))
#    return z 
#
#        