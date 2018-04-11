# -*- coding: utf-8 -*-
"""
Created on Tue Apr 10 17:15:26 2018

@author: user
"""

import pymysql
import pandas as pd 
import time
from pandas import Series, DataFrame


def get_premiumfromdb():
   
    # MySQL Connection 연결
    conn = pymysql.connect(host='127.0.0.1', user='rabbit', password='rabbit',db='kpsml', charset='utf8')
     
    # Connection 으로부터 Cursor 생성
    curs = conn.cursor()
    # SQL문 실행
    sql= "select * from premium limit 5;"
    #sql = "select address from contact where member_no in (select member_no from alarm where percent="+predict_percent+" and type=1) and type=1 and certification=1"
    curs.execute(sql)
    
    # 데이타 Fetch
    rows = curs.fetchall()
    rows=list(rows)
    df={'5min_0':[],'premium':[]}
        
    for i,j in enumerate(rows):
        df['5min_0'].append(rows[i][0])
        df['premium'].append(rows[i][1])
       
    # Connection 닫기
    conn.close()
    return df


data=get_premiumfromdb()
print(data)  


"""
def getData():
    data=pd.read_csv("data/datainputset.csv")
    #data[-1]에 있는 값 찾기 -> string 값으로 변환하기
    timestamp='1523338880'
    dbdata=get_premiumfromdb('1523338880')
    dbdata=pd.DataFrame(dbdata)
    data= pd.concat([data, dbdata]) 
    return data
  

def get_premiumfromdb(timestamp):
   
    # MySQL Connection 연결
    conn = pymysql.connect(host='10.1.43.149', user='rabbit', password='rabbit',db='kps', charset='utf8')
     
    # Connection 으로부터 Cursor 생성
    curs = conn.cursor()
    timestamp='1523338880' 
    # SQL문 실행
    sql= "select * from premium where timestamp>"+timestamp+";"
    #sql = "select address from contact where member_no in (select member_no from alarm where percent="+predict_percent+" and type=1) and type=1 and certification=1"
    curs.execute(sql)
    
    # 데이타 Fetch
    rows = curs.fetchall()
    rows=list(rows)
    df={'5min_0':[],'premium':[]}
        
    for i,j in enumerate(rows):
        df['5min_0'].append(rows[i][0])
        df['premium'].append(rows[i][1])
       
    # Connection 닫기
    conn.close()
    return df


""""
    