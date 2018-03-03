#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 22 21:50:45 2018

@author: amajidsinar
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 20 23:30:12 2018

@author: amajidsinar
"""

import numpy as np
    

class Knn():
    def __init__(self, k, dist='l1'):
        avDist = ['l1', 'manhattan']
        if dist not in avDist:
            pass
        self.k = k
        self.dist = dist
    
    def fit(self,data_known,label_known):
        self.data_known = data_known
        self.label_known = label_known
        
    def L1_distance(self):
        diff = self.data_known - self.data_unknown.reshape((self.data_unknown.shape[0],1,self.data_unknown.shape[1]))
        return (diff**2).sum(2)
    
    def manhattan(self):
        diff = self.data_known - self.data_unknown.reshape((self.data_unknown.shape[0],1,self.data_unknown.shape[1]))
        return np.abs(diff).sum(2)
    
    def predict(self, data_unknown):
        self.data_unknown = data_unknown
        #sort label
        if self.dist == 'euc':
            dist_index = np.argsort(self.L2_distance())
        else:
            dist_index = np.argsort(self.manhattan())
        
        label = self.label_known[dist_index]
        #only pick until kth index
        label = label[:,:self.k]
        #return the mode
        label_predict = []
        for i in range(self.data_unknown.shape[0]):
            values,counts =  np.unique(label[i], return_counts=True)
            ind = np.argmax(counts)
            label_predict.append(values[ind])
        return label_predict
    
        
import random
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
digits = datasets.load_digits()

data_known = digits.data
label_known = digits.target

data_known,label_known,data_unknown,label_unknown = split(data_known,label_known,0.8)

val_accuracy = []
for i in np.arange(1,40):
    knn = Knn(i)
    knn.fit(data_known,label_known)
    label_predict = knn.predict(data_unknown)
    performance = np.mean(label_predict == label_unknown)
    val_accuracy.append(performance)

import matplotlib.pyplot as plt
plt.title('accuracy vs k plot of digit recognition')
plt.xlabel('k')
plt.ylabel('accuracy')
plt.plot(val_accuracy)




