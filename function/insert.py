# -*- coding: utf-8 -*-
"""
Created on Thu Apr  5 11:16:54 2018

@author: user
"""

import pymysql


    
def insertpremium(time_premiumdata) :
    
    # MySQL Connection 연결
    conn = pymysql.connect(host='35.187.205.146', user='rabbit', password='rabbit',db='kpsml', charset='utf8')
     
    # Connection 으로부터 Cursor 생성
    curs = conn.cursor()
    
#    print(type(time_premiumdata))
#    print(type(time_premiumdata['premium']))
#    print(time_premiumdata['premium'])
    # SQL문 실행
    sql="insert into premium values (%s,%s)"
    curs.execute(sql, (time_premiumdata['timestamp'], time_premiumdata['premium']))
    
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