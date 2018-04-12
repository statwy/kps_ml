# -*- coding: utf-8 -*-
"""
Created on Thu Apr  5 11:16:54 2018

@author: user
"""

import pymysql

def truncate() :
    # MySQL Connection 연결
    print("bitpred지우기 시작")
    conn = pymysql.connect(host='45.119.144.83 ', user='rabbit1', password='rabbit1',db='kps', charset='utf8')    
    # Connection 으로부터 Cursor 생성
    curs = conn.cursor()    
    sql="truncate table bitpred"
    curs.execute(sql)    
    conn.commit()
    conn.close()

def insertpremium2(df) :
    
    # MySQL Connection 연결
    conn = pymysql.connect(host='45.119.144.83 ', user='rabbit1', password='rabbit1',db='kps', charset='utf8')
     
    # Connection 으로부터 Cursor 생성
    curs = conn.cursor()
    
    print("bitpred삽입 시작")
    for i in range(0,len(df['timestamp'])):
        sql="insert into bitpred values (%s,%s)"
        curs.execute(sql, (df['timestamp'][i], df['premium'][i]))    
    conn.commit()
    conn.close()


