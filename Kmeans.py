# -*- coding: utf-8 -*-
import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

def getClosePattern(data, n):
    loc = tuple(range(1, len(data) - n, 3))
    
    column = [str(e) for e in range(1, (n+1))]
    df = pd.DataFrame(columns=column)
    
    for i in loc:       
        pt = data['premium'].iloc[i:(i+n)].values
        pt = (pt - pt.mean()) / pt.std()
        df = df.append(pd.DataFrame([pt],columns=column, index=[data.index[i+n]]), ignore_index=False)
        
    return df

data=pd.read_csv("data/data_refined.csv")
ft = getClosePattern(data, n=100)

# Pattern 몇 개를 확인해 본다
#x = np.arange(20)
#plt.plot(x, ft.iloc[0])
#plt.plot(x, ft.iloc[10])
#plt.plot(x, ft.iloc[50])
#plt.show()
#print(ft.head())

# K-means 알고리즘으로 Pattern 데이터를 8 그룹으로 분류한다 (k = 8)
k = 8
km = KMeans(n_clusters=k, init='random', n_init=10, max_iter=300, tol=1e-04, random_state=0)
km = km.fit(ft)
y_km = km.predict(ft)
ft['cluster'] = y_km
ft.to_csv("data/kmeansdata.csv")
## Centroid pattern을 그린다
#fig = plt.figure(figsize=(10, 6))
#for i in range(k):
#    s = 'pattern-' + str(i)
#    p = fig.add_subplot(2,4,i+1)
#    p.plot(km.cluster_centers_[i,:], color="rbgkmrbg"[i])
#    p.set_title('Cluster-' + str(i))
#
#plt.tight_layout()
#plt.show()
#
## cluster = 0 인 패턴 몇 개만 그려본다
#cluster = 0
#plt.figure(figsize=(8, 5))
#p = ft.loc[ft['cluster'] == cluster]
#for i in range(5):
#    plt.plot(x,p.iloc[i][0:20])
#    
#plt.title('Cluster-' + str(cluster))
#plt.show()



