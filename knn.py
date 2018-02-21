#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 13 18:52:28 2018

@author: amajidsinar
"""

from sklearn import datasets
import matplotlib.pyplot as plt
import numpy as np


iris = datasets.load_iris()

dataset = iris.data
# only take 0th and 1th column for X
X_train = iris.data[:,:2]
# y
y_train = iris.target

# the hard part
# so matplotlib does not readily support labeling based on class
# but we know that one of the feature of plt is that a plt call would give those set of number
# the same color
category = np.unique(y_train)
for i in category:
    upper = np.max(np.where(y_train == i))
    lower = np.min(np.where(y_train == i))
    plt.scatter(X_train[lower:upper,0],X_train[lower:upper,1],label=i)

# Unknown class of a point
X_test = np.array([[5.7,3.3],[5.6,3.4]])
plt.scatter(X_test[:,0],X_test[:,1], label='?')
plt.legend()
#-------------
# Euclidean Distance
diff = X_train - X_test.reshape(len(X_test),1,len(X_test))
distance = (diff**2).sum(2)
distance = distance.reshape(len(X_test),1,len(X_train))


#return sorted index of distance
dist_index = np.argsort(distance)
label = y_train[dist_index]

k = 5

#keep the rank

label = label[:,:,:k]

result = []    
for i in range(len(X_test)):
    values,counts =  np.unique(label[i], return_counts=True)
    ind = np.argmax(counts)
    result.append(values[ind])

result





    
    




