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


time_premiumdata={'timestamp_p':[],'premium':[]}
start_time_m=datetime.datetime.now()
i=0
while True :
    i+=1    
    time_premiumdata['timestamp_p'].append(int(time.time()))
    time_premiumdata['premium'].append(crawling())  
    time_for_logic_m=datetime.datetime.now()-start_time_m
    if time_for_logic_m.total_seconds()>500 :   
        time_premiumdata=pd.DataFrame(time_premiumdata)
        time_premiumdata.to_csv('data/premium'+str(i)+'.csv')
        start_time_m=datetime.datetime.now()
        time_premiumdata={'timestamp_p':[],'premium':[]}
    print(time_premiumdata)
    