# -*- coding: utf-8 -*-
"""
Created on Mon Apr  2 16:03:30 2018

@author: user
"""


import os
import subprocess
from hdfs import InsecureClient

def savetohadoop(filename):
    os.chdir('/home/rabbit/hadoop2/')
    subprocess.call('./bin/hdfs dfs -ls /test', shell=True)
    file=filename
    subprocess.call('./bin/hdfs dfs -put /home/rabbit/data_anal/kps_ml/data/'+file+'.csv /user/hdfs/test', shell=True)

def savetohadoop_d(dataframe,filename):
    client_hdfs = InsecureClient('http://http://10.1.43.149:50070')
    with client_hdfs.write('/user/hdfs/test/'+filename+'.csv', encoding = 'utf-8') as writer:
        dataframe.to_csv(writer)                                
       