# -*- coding: utf-8 -*-
"""
Created on Mon Apr  2 16:03:30 2018

@author: user
"""


import os
import subprocess
from hdfs import InsecureClient

def savetohadoop(filename):
    #하둡 경로로 이동하기
    os.chdir('/home/rabbit/hadoop2/')
    subprocess.call('./bin/hdfs dfs -ls /test', shell=True)
    #python이 실행중인 경로명
    pwd_python=subprocess.call('pwd', shell=True)
    file=filename
    subprocess.call('./bin/hdfs dfs -put '+pwd_python+'/data/'+file+'.csv /user/hdfs/coindata1', shell=True)


def savetohadoop_d(dataframe,filename):
    client_hdfs = InsecureClient('http://http://10.1.43.149:50070')
    with client_hdfs.write('/user/hdfs/coindata2/'+filename+'.csv', encoding = 'utf-8') as writer:
        dataframe.to_csv(writer)                                
       