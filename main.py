# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from function.crawling_for_mail  import crawling
from function.mail_func import mail


mail_address=['jwy627@naver.com','jwy627wywy@naver.com','jwy627@korea.ac.kr']
premium=[]
i=0
while True :
    i+=1
    premium=append_maxsize(premium,crawling(),1000)
    upper_bound,lower_bound=bollingerband(premium)
    crawling()
    if i==300 :
        break



#mail('test',mail_address)


plt.plot(upper_bound)
plt.plot(premium)
plt.plot(lower_bound)    




def bollingerband(x) :
    x=pd.DataFrame(x)
    mva=x.rolling(window=100).mean()
    mvstd=x.rolling(window=100).std()
    upper_bound=mva+2*mvstd
    lower_bound=mva-2*mvstd
    return upper_bound, lower_bound


def append_maxsize(x,y,z) :
    if len(x)<z :
        x.append(y)
    else :
        x.pop(0)
        x.append(y) 
    return x



#####################################################################
# 밑으로 아직 덜 짰음 ...
def gener_percent(x,y) :
    z={}
    for i in range(x,y+1) :
       print('%d') %i
        #x['p_%d']=0  %i 
    return z 
        



def flag() :
    for i in range(-16,60) :
        global p_%d
   
   
for i in range (1,10) :
    print('%d') %i
   
   
