# -*- coding: utf-8 -*-
"""
Created on Tue Apr 10 14:07:29 2018

@author: user
"""

import pymysql

def get_premium(days) :
    
    days=str(days)
    # MySQL Connection 연결
    
    conn = pymysql.connect(host='35.187.205.146', user='rabbit', password='rabbit',db='kpsml', charset='utf8')
     
    # Connection 으로부터 Cursor 생성
    curs = conn.cursor()
     
    # SQL문 실행
    sql= "select * from premium order by timestamp DESC limit "+days+";"
    #sql = "select address from contact where member_no in (select member_no from alarm where percent="+predict_percent+" and type=1) and type=1 and certification=1"
    curs.execute(sql)
    
    # 데이타 Fetch
    rows = curs.fetchall()
    rows=list(rows)
    df={'timestamp':[],'premium':[]}
    
    for i,j in enumerate(rows):
        df['timestamp'].append(rows[i][0])
        df['premium'].append(rows[i][1])
       
    # Connection 닫기
    conn.close()
    return df

#get_memberlist(3.0)

get_premium(10)

rows, premiumlist = get_premium(10)
print(rows)

type(rows)
