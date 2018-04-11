# -*- coding: utf-8 -*-
"""
Created on Thu Apr  5 11:16:54 2018

@author: user
"""

import pymysql
#x=timestamp,y=premium


#def truncate() :
#    # MySQL Connection 연결
#    conn = pymysql.connect(host='10.1.43.149', user='rabbit', password='rabbit',db='kps', charset='utf8')     
#    # Connection 으로부터 Cursor 생성
#    curs = conn.cursor()    
#    sql="truncate table bitpred"
#    curs.execute(sql)    
#    conn.commit()
#    conn.close()
    
def truncate() :
    # MySQL Connection 연결
    conn = pymysql.connect(host='45.119.144.83 ', user='rabbit1', password='rabbit1',db='kps', charset='utf8')    
    # Connection 으로부터 Cursor 생성
    curs = conn.cursor()    
    sql="truncate table bitpred"
    curs.execute(sql)    
    conn.commit()
    conn.close()

#
#def insertpremium(df) :
#    
#    # MySQL Connection 연결
#    conn = pymysql.connect(host='10.1.43.149', user='rabbit', password='rabbit',db='kps', charset='utf8')
#     
#    # Connection 으로부터 Cursor 생성
#    curs = conn.cursor()
#    
#    for i in range(0,len(df['timestamp'])):
#        sql="insert into bitpred values (%s,%s)"
#        curs.execute(sql, (df['timestamp'][i], df['premium'][i]))    
#    conn.commit()
#    conn.close()


def insertpremium(df) :
    
    # MySQL Connection 연결
    conn = pymysql.connect(host='45.119.144.83 ', user='rabbit1', password='rabbit1',db='kps', charset='utf8')
     
    # Connection 으로부터 Cursor 생성
    curs = conn.cursor()
    
    for i in range(0,len(df['timestamp'])):
        sql="insert into bitpred values (%s,%s)"
        curs.execute(sql, (df['timestamp'][i], df['premium'][i]))    
    conn.commit()
    conn.close()



    
#    myDict={'wifi':'11', 'wifi_status':'1L'}
#    columns=','.join(myDict.key())
#    placeholders=','.join([%s]*len(myDict))
#    sql="insert into %s 
#    
#    # SQL문 실행
#    sql = "select address from contact where member_no in (select member_no from alarm where percent="+predict_percent+" and type=1) and type=1 and certification=1"
#    curs.execute(sql)
#    
#    # 데이타 Fetch
#    rows = curs.fetchall()
#    rows=list(rows)
#    memberlist=[]
#    
#       
#    # Connection 닫기
#    conn.close()
#    return memberlist