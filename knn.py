#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 13 18:52:28 2018

@author: amajidsinar
"""

from sklearn import datasets
import matplotlib.pyplot as plt
import numpy as np
plt.style.use('seaborn-white')


iris = datasets.load_iris()

dataset = iris.data
# only take 0th and 1th column for X
point_known = iris.data[:,:2]
# y
label_known = iris.target

# the hard part
# so matplotlib does not readily support labeling based on class
# but we know that one of the feature of plt is that a plt call would give those set of number
# the same color
category = np.unique(label_known)
for i in category:
    upper = np.max(np.where(label_known == i))  
    lower = np.min(np.where(label_known == i))
    plt.scatter(point_known[lower:upper,0],point_known[lower:upper,1],label=i)

# Unknown class of a point
point_unknown = np.array([[5.7,3.3],[5.6,3.4]])
plt.scatter(point_unknown[:,0],point_unknown[:,1], label='?')
plt.legend()
#-------------
# Euclidean Distance
diff = point_known - point_unknown.reshape(len(point_unknown),1,len(point_unknown))
distance = (diff**2).sum(2)
distance = distance.reshape(len(point_unknown),1,len(point_known))


#return sorted index of distance
dist_index = np.argsort(distance)
label = label_known[dist_index]

k = 5

#keep the rank

label = label[:,:,:k]

y_unknown = []    
for i in range(len(point_unknown)):
    values,counts =  np.unique(label[i], return_counts=True)
    ind = np.argmax(counts)
    y_unknown.append(values[ind])







    
    




