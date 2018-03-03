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
import matplotlib.pyplot as plt
plt.style.use("seaborn-white")

class Model:
    def evaluate(self):
        print(self.predict())
        print(self.label_unknown)
        return np.mean(self.predict() == self.label_unknown)

class Knn(Model):
   
    def __init__(self, k, dist='euc'):
        avDist = ['euc', 'man']
        if dist not in avDist:
            pass
        self.k = k
        self.dist = dist
        
    
    def fit(self,data_known,label_known,data_unknown,label_unknown):
        self.data_known = data_known
        self.label_known = label_known
        self.data_unknown = data_unknown
        self.label_unknown = label_unknown
        
    def L1_distance(self):
        diff = self.data_known - self.data_unknown.reshape((self.data_unknown.shape[0],1,self.data_unknown.shape[1]))
        return (diff**2).sum(2)
    
    def manhattan(self):
        diff = self.data_known - self.data_unknown.reshape((self.data_unknown.shape[0],1,self.data_unknown.shape[1]))
        return np.abs(diff).sum(2)
        
        
    def L2_distance(self):
        diff = self.data_known - self.data_unknown.reshape((self.data_unknown.shape[0],1,self.data_unknown.shape[1]))
        return np.sqrt((diff**2).sum(2))
    
    
    def predict(self):
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
iris = datasets.load_iris()

data_known = iris.data[:,:2]
label_known = iris.target

data_known,label_known,data_unknown,label_unknown = split(data_known,label_known,0.9)

knn1 = Knn(5,dist='eu')
#knn1.fit(data_known,label_known,data_unknown,label_unknown)
#performance = knn1.evaluate()



