# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from function.crawling_for_mail  import crawling
from function.mail_func import mail
from function.get_memberlist import get_memberlist
import math


#mail_address 가져올 때 사용할 함수 짜야함
#mail_address=['jwy627wywy@naver.com']


# 초기 데이터 생성용 로직
j=0
premium=[]
for j in range(0,100) : 
    premium.append(crawling())
    j+=1
upper_bound,lower_bound=bollingerband(premium,100)

print('##### 메일 보내는 로직 시작 #####')
# 메일 보내는 로직 (mail 함수 제목 주는거 고쳐야함)
i=0
gener_percentFlag(-10,10)
while True :
    i+=1
    premium=append_maxsize(premium,crawling(),1000)
    upper_bound,lower_bound=bollingerband(premium,100)
    
    if not lower_bound[-1] < crawling() < upper_bound[-1]:
        percent_flag['Boll']=1
        
        print( 'Boll 1 이어야돼', percent_flag['Boll'] )
        
    if not math.floor(premium[-2]*100)==math.floor(premium[-1]*100) :
        temp=premium_int(premium[-2],premium[-1])
        temp_premium='p_%d' %temp
       
        if percent_flag['Boll']==1 or percent_flag[temp_premium]==1  :
            mail_address=get_memberlist(temp)
            print(mail_address)
            #mail_address=['jwy627wywy@naver.com']
        
            mail('test',mail_address)
            
            print('percent_flag[temp_premium] 몇이야', percent_flag[temp_premium])
            
        setup_percentFlag(-10,10,premium_int(premium[-2],premium[-1]))
        
        print( 'SET Boll 0 이어야돼', percent_flag['Boll'] )
        print('SET percent_flag[temp_premium] 몇이야(0이어야돼)', percent_flag[temp_premium])
    if i==200 :
        break

 


# bollinger band test plot
plt.plot(upper_bound)
plt.plot(premium)
plt.plot(lower_bound)    



#함수, 나중에 다른 폴더로 뺄 계획 
##############################################################################

def bollingerband(x,y) :
    x=pd.DataFrame(x)
    mva=x.rolling(window=y).mean()
    mvstd=x.rolling(window=y).std()
    upper_bound=mva+5*mvstd
    lower_bound=mva-5*mvstd
    return list(upper_bound[0]), list(lower_bound[0])


def append_maxsize(x,y,z) :
    if len(x)<z :
        x.append(y)
    else :
        x.pop(0)
        x.append(y) 
    return x


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

    
    
def premium_int(x,y) : 
    z= max(math.floor(x*100),math.floor(y*100))
    return z 

        