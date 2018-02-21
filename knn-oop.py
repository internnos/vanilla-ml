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
    
    def __init__(self,X_train,y_train,X_test,k):
        self.X_train = X_train
        self.y_train = y_train
        self.X_test = X_test
        self.k = k
    def plot(self,xlabel,ylabel):
        category = np.unique(y_train)
        for i in category:
            upper = np.max(np.where(self.y_train == i))
            lower = np.min(np.where(self.y_train == i))
            plt.scatter(self.X_train[lower:upper,0],self.X_train[lower:upper,1],label=i)
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.scatter(self.X_test[:,0],self.X_test[:,1], label='?')
        plt.legend()
    def predict(self):
        #euclidian distance
        diff = self.X_train - self.X_test.reshape(len(self.X_test),1,len(self.X_test))
        distance = (diff**2).sum(2)
        distance = distance.reshape(len(self.X_test),1,len(self.X_train))
        #sort label
        dist_index = np.argsort(distance)
        label = self.y_train[dist_index]
        #only pick until kth index
        label = label[:,:,:self.k]
        #return the mode
        result = []
        for i in range(len(self.X_test)):
            values,counts =  np.unique(label[i], return_counts=True)
            ind = np.argmax(counts)
            result.append(values[ind])
        return result
    
        
#load iris
iris = datasets.load_iris()
dataset = iris.data

#set input
X_train = iris.data[:,:2]
y_train = iris.target
X_test = np.array([[5.7,3.3],[5.6,3.4]])

#create instance of knn

knn_1 = Knn(X_train,y_train,X_test,1)
knn_2 = Knn(X_train,y_train,X_test,2)
knn_3 = Knn(X_train,y_train,X_test,3)
knn_4 = Knn(X_train,y_train,X_test,4)
knn_5 = Knn(X_train,y_train,X_test,5)
knn_6 = Knn(X_train,y_train,X_test,6)
knn_7 = Knn(X_train,y_train,X_test,7)
knn_8 = Knn(X_train,y_train,X_test,8)
knn_9 = Knn(X_train,y_train,X_test,9)
knn_10 = Knn(X_train,y_train,X_test,10)


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

knn_1.plot('sepal length', 'sepal width')
