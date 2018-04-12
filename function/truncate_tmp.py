# -*- coding: utf-8 -*-
"""
Created on Thu Apr 12 11:24:03 2018

@author: user
"""
import pymysql


print("bitpred지우기 시작")
conn = pymysql.connect(host='45.119.144.83 ', user='rabbit1', password='rabbit1',db='kps', charset='utf8')    
# Connection 으로부터 Cursor 생성
curs = conn.cursor()    
sql="truncate table bitpred"
curs.execute(sql)    
conn.commit()
conn.close()