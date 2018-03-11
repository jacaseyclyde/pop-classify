# -*- coding: utf-8 -*-
"""
K-Means Implementation
"""

import time
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split

cdict = {'O':'blue','B':'lightskyblue','A':'white','F':'lightyellow',
             'G':'yellow','K':'orange','M':'red','T':'brown','L':'saddlebrown',
             'C':'black','W':'purple'}
###-----------------------------------------------------------------------###
#data creation 

print("Importing data...")
u, g, r, i, z = np.genfromtxt('data.csv', delimiter=',', skip_header=2,usecols=(0,1,2,3,4)).T
subclass = np.genfromtxt('data.csv', delimiter=',',skip_header=2,usecols=5,dtype=str)
print("Import complete!")

colordata = np.array([u-g, g-r, r-i, i-z]).T #, u-r, u-i, u-z, g-i, g-z, r-z
    
###-----------------------------------------------------------------------###
#outlier deletion 

i_extr = np.where(np.logical_or.reduce(np.abs(colordata) > 100, axis=1))
    
colordata = np.delete(colordata, i_extr, axis=0)
subclass = np.delete(subclass, i_extr, axis=0)
###-----------------------------------------------------------------------###
#subclass array 

stellar_class = []
for c in subclass:
   stellar_class.append(c[0])
stellar_class = np.array(stellar_class)
###----------------------------------------------------------------------###
#Training and Testing set creation
   
clr_train, clr_test, cls_train, cls_test = train_test_split(colordata, stellar_class,
                                                                test_size=.5, random_state=0)

###---------------------------------------------------------------------###
#K-means clustering
#kmeans does the training, clustering the test data
#kmeans.predict does the testing, predicting which cluster each new datapoint falls into
#kclus array stores predictions of various cluster counts (5-11), all using same data

def kmeantest(clr_train,clr_test,cl_count):
    kclus = np.zeros([cl_count-5,len(clr_test)]).T
    for i in range(5,cl_count+1):
        kmeans = KMeans(n_clusters=i).fit(clr_train)
        kclus[:,i-6] = kmeans.predict(clr_test)
        print('Test with cluster count of {0} is complete'.format(i))
    return kclus

#TODO: Match distribution of stellar classes to clusters to determine predictive and 
# clustering strength of k-means algorithm