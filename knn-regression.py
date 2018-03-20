#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 19 18:46:26 2018

@author: amajidsinar
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 13 18:52:28 2018

@author: amajidsinar
"""

import numpy as np
import matplotlib.pyplot as plt
plt.style.use('seaborn-white')

import random
random.seed(123)

def split(data_known,label_known,training_percentage):
    #data_set and label is static
    data_set = data_known
    label = label_known
    #take percentage*len(data)
    index = random.sample(range(len(data_known)),int(training_percentage*len(data_known)))
    data_known = data_set[index]
    label_known =  label[index]
    data_unknown = np.delete(data_set, index, axis=0)
    label_unknown = np.delete(label, index, axis=0)
    return (data_known,label_known,data_unknown,label_unknown)

#load iris
from sklearn import datasets
boston = datasets.load_boston()

#data_known = np.array((boston.data[:,0]), ndmin=2).T
data_known = boston.data
label_known = boston.target

data_known,label_known,data_unknown,label_unknown = split(data_known,label_known,0.8)


diff = data_known - data_unknown.reshape(data_unknown.shape[0],1,data_unknown.shape[1])
distance = (diff**2).sum(2)
dist_index = np.argsort(distance)
label = label_known[dist_index]

k = 6
label = label[:,:k]

comp_mean = []
for i in range(label.shape[0]):
    comp_mean.append(np.mean(label[i]))

mse = np.sqrt(np.sum(label_unknown - comp_mean)**2)


from sklearn import neighbors
knn = neighbors.KNeighborsRegressor(6)
y_ = knn.fit(data_known, label_known).predict(data_unknown)


mse_scikit = np.sqrt(np.sum(label_unknown - y_)**2)