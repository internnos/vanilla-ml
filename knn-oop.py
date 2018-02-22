#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 20 23:30:12 2018

@author: amajidsinar
"""

class Knn:
    import numpy as np
    from sklearn import datasets
    from scipy import stats
    import matplotlib.pyplot as plt
    plt.style.use("seaborn-white")
    
    def __init__(self,data_known,label_known,data_unknown,k):
        self.data_known = data_known
        self.label_known = label_known
        self.data_unknown = data_unknown
        self.k = k
    def plot(self,title,xlabel,ylabel):
        category = np.unique(label_known)
        for i in category:
            upper = np.max(np.where(self.label_known == i))
            lower = np.min(np.where(self.label_known == i))
            plt.scatter(self.data_known[lower:upper,0],self.data_known[lower:upper,1],label=i)
        plt.title(title)
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.scatter(self.data_unknown[:,0],self.data_unknown[:,1], label='?')
        plt.legend()
    def predict(self):
        #euclidian distance
        diff = self.data_known - self.data_unknown.reshape(len(self.data_unknown),1,len(self.data_unknown))
        distance = (diff**2).sum(2)
        distance = distance.reshape(len(self.data_unknown),1,len(self.data_known))
        #sort label
        dist_index = np.argsort(distance)
        label = self.label_known[dist_index]
        #only pick until kth index
        label = label[:,:,:self.k]
        #return the mode
        label_unknown = []
        for i in range(len(self.data_unknown)):
            values,counts =  np.unique(label[i], return_counts=True)
            ind = np.argmax(counts)
            label_unknown.append(values[ind])
        return label_unknown
    
        
#load iris
iris = datasets.load_iris()
dataset = iris.data

#set input
data_known = iris.data[:,:2]
label_known = iris.target
data_unknown = np.array([[5.7,3.3],[5.6,3.4]])

#create instance of knn

knn_1 = Knn(data_known,label_known,data_unknown,1)
knn_2 = Knn(data_known,label_known,data_unknown,2)
knn_3 = Knn(data_known,label_known,data_unknown,3)
knn_4 = Knn(data_known,label_known,data_unknown,4)
knn_5 = Knn(data_known,label_known,data_unknown,5)
knn_6 = Knn(data_known,label_known,data_unknown,6)
knn_7 = Knn(data_known,label_known,data_unknown,7)
knn_8 = Knn(data_known,label_known,data_unknown,8)
knn_9 = Knn(data_known,label_known,data_unknown,9)
knn_10 = Knn(data_known,label_known,data_unknown,10)


print(knn_1.predict())
print(knn_2.predict())
print(knn_3.predict())
print(knn_4.predict())
print(knn_5.predict())
print(knn_6.predict())
print(knn_7.predict())
print(knn_8.predict())
print(knn_9.predict())
print(knn_10.predict())

knn_1.plot('Iris dataset','sepal length', 'sepal width')
