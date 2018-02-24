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

class Knn:
    import numpy as np
    from scipy import stats
    import matplotlib.pyplot as plt
    import random
    plt.style.use("seaborn-white")
    
    def split(self,data_known,label_known,training_percentage):
        import random
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
    
    def plot(self,title,xlabel,ylabel):
        category = np.unique(label_known)
        for i in category:
            upper = np.max(np.where(label_known == i))
            lower = np.min(np.where(label_known == i))
            plt.scatter(data_known[lower:upper,0],data_known[lower:upper,1],label=i)
        plt.title(title)
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.scatter(data_unknown[:,0],data_unknown[:,1], label='?')
        plt.legend()
        
    def euclidean_distance(self,data_known,data_unknown):
        diff = data_known - data_unknown.reshape((data_unknown.shape[0],1,data_unknown.shape[1]))
        distance = (diff**2).sum(2)
        distance = distance.reshape(data_unknown.shape[0],1,data_known.shape[0])
        return distance
    
    def predict(self,data_known,data_unknown,label_known,k):
        #sort label
        dist_index = np.argsort(self.euclidean_distance(data_known,data_unknown))
        label = label_known[dist_index]
        #only pick until kth index
        label = label[:,:,:k]
        #return the mode
        label_unknown = []
        for i in range(len(data_unknown)):
            values,counts =  np.unique(label[i], return_counts=True)
            ind = np.argmax(counts)
            label_unknown.append(values[ind])
        return label_unknown

    
        
#load iris
import numpy as np
from sklearn import datasets
iris = datasets.load_iris()

#data_known = iris.data[:,:2]
#label_known = iris.target
#data_unknown = np.array([[5.7,3.3],[5.6,3.4]])

data_known = iris.data[:,:2]
label_known = iris.target

knn = Knn()
data_known, label_known, data_unknown, label_unknown = knn_1.split(data_known,label_known,0.4)
#set input
#first = trainingAndTest(iris.data, iris.target, 0.4)
#data_known,label_known,data_unknown,label_unknown = first()
predict1 = knn.predict(data_known, data_unknown, label_known,1)
predict2 = knn.predict(data_known, data_unknown, label_known,2)
predict3 = knn.predict(data_known, data_unknown, label_known,3)
predict4 = knn.predict(data_known, data_unknown, label_known,4)
predict5 = knn.predict(data_known, data_unknown, label_known,5)



