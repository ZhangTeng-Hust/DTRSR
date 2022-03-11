# -*- coding: utf-8 -*-
"""
Created on 14:55,2021/11/26
@author: ZhangTeng
Seed Technology
"""

import numpy as np
from sklearn.cluster import KMeans
import copy

def E_distance_2D(x, y):

    distance = y - x
    return distance

def E_distance_1D(x, y):
    
    dist = np.linalg.norm(y - x)
    return dist

def Find_same(x, Q ,index_num):

    distance = []
    for i in range(len(Q)):
        distance.append(E_distance_1D(x, Q[i, :]))
    min_index = np.argmin(distance, axis=0)

    while min_index in index_num:
        distance[min_index] = 1e100   
        # Here you can set the threshold value according to the actual situation, theoretically the larger the better
        min_index = np.argmin(distance,axis=0)
    index_num.append(min_index)
    return Q[min_index, :]

def CLUSTER(Xs, Xt): 
   
   
    index_num = []
    number = np.shape(Xt)[0]
    estimator = KMeans(n_clusters=number, init='k-means++')
    Xs = Xs.cpu().detach().numpy()
    estimator.fit(Xs)
    label_pred = estimator.labels_  
    centroids = estimator.cluster_centers_ 
    numSamples = len(Xs)
    new_Xt = []
   
    for find in range(len(Xt)):  
        new_Xt.append(Find_same(centroids[find, :], Xt.cpu().detach().numpy(),index_num))
    new_Xt = np.array(new_Xt)
    new_Xs = copy.deepcopy(Xs)  
    Residual = np.zeros((np.shape(new_Xs)[0],np.shape(new_Xs)[1]))
    for i in range(numSamples):
        distance = E_distance_2D(centroids[label_pred[i]], Xs[i, :])  
        new_Xs[i, :] = new_Xt[label_pred[i], :] + distance
        Residual[i,:] = distance

    return new_Xs, label_pred, Residual
