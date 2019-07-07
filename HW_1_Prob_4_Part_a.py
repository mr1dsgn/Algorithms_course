# -*- coding: utf-8 -*-
"""
@author: AhmedShokry
"""
#Importing the libraries
from __future__ import division
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math
from numpy import ndarray


#Gaussian
def gaussian(x, mean, cov):
    xm = np.absolute(np.array(x) - np.array(mean))
    xm=np.reshape((x-mean), (-1, 1))
    px=1/(math.pow(2.0*math.pi, 2))*1/math.sqrt(np.linalg.det(cov))*math.exp(-(np.dot(np.dot(xm.T, np.linalg.inv(cov)), xm))/2)
    return px

def sigmoid(a):
    sig = 1 / (1 + np.exp(-a))
    return sig

from sklearn.metrics import mean_squared_error
#def calculateError(X, Y, weight):
#    Ypredict = np.matmul(X, weight) #y predict
#    return err(Y, Ypredict)

#Importing the dataset
dataset=pd.read_csv('spambase.data.csv')
X = dataset.iloc[:, :-1].values         # Data
y = dataset.iloc[:, -1].values          # Target


#split training set and testing set
from sklearn.cross_validation import train_test_split
xtrain, xtest, ytrain, ytest = train_test_split(X, y, test_size=0.20, random_state = 1)# add random state

class1 = []  #spam
class2 = []  #No spam

for n in range(ytrain.shape[0]):
    if (y[n]==1):
        class1.append(xtrain[n,:])
    elif(y[n]==0):
        class2.append(xtrain[n,:])
   
class1 = np.array(class1)
class2 = np.array(class2)


#count=np.zeros(shape=(150,1))
#t_assigned=np.zeros(shape=(150, 3))
#cov=np.zeros(shape=(3, 4, 4))
#mean=np.zeros(shape=(3, 4))

#compute means for each class
mean1=class1.mean(axis=0)
mean2=class2.mean(axis=0)

#compute covariance matrices, such that the columns are variables and rows are observations of variables
cov1=np.cov(class1, rowvar=0)
cov2=np.cov(class2, rowvar=0)

##compute gaussian likelihood functions p(x|Ck) for each class
#for i in xrange(len(dataset)):
#    px1=(1/2.0)*gaussian(dataset[i], mean1, cov1)
#    px2=(1/2.0)*gaussian(dataset[i], mean2, cov2)
#    m=np.max([px1, px2])
 #compute posterior probability p(Ck|x) assuming that p(x|Ck) is gaussian and the entire expression is wrapped by sigmoid function 

N1 = class1.shape[0]
N2 = class2.shape[0]
pi = N1 / (N1+N2)

S = N1/(N1+N2)*cov1 + N2/(N1+N2)*cov2

w = np.matmul(np.linalg.inv(S), mean1)
w0 = -0.5*np.matmul(mean1.transpose(), w) + pi

pcTrain = sigmoid(np.matmul(xtrain, w) + w0)
pcTest = sigmoid(np.matmul(xtest, w) + w0)
mean_squared_error(ytrain, pcTrain)
mean_squared_error(ytest, pcTest)
# #assign p(Ck|x)=1 if p(Ck|x)>>p(Cj|x) for all j!=k
#     if pc1>pc2 : t_assigned[i][0]=0
#     elif pc2>pc1: t_assigned[i][1]=1
#     
# #count the number of misclassifications
#     for j in xrange(2):
#         if t[i][j]-t_assigned[i][j]!=0: count[i]=1
#
# cov=[cov1, cov2]
# mean=[mean1, mean2]
#
# t1=np.zeros(shape=(len(class1), 1))
# t2=np.zeros(shape=(len(class2), 1))
# 
# for i in xrange(len(data)):
#     for j in xrange(len(class1)):
#         if t_assigned[i][0]==1: t1[j]=1
#         elif t_assigned[i][1]==1: t2[j]=2

 plt.plot(t1, "bo", label="Class 1")
 plt.plot(t2, "go", label="Class 0")
 
 plt.legend()
 plt.show()





