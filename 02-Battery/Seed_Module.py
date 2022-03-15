# -*- coding: utf-8 -*-
"""
Created on 14:55,2021/09/13
@author: ZhangTeng
Seed Technology
"""

import numpy as np
from sklearn.cluster import KMeans
import copy

def E_distance_2D(x, y):
    """
    x 表示 center；
    y 表示 sample；
    distance 表示当前聚类中样本和聚类中心之间，各个维度差值组成的矩阵；
    """
    distance = y - x
    return distance

def E_distance_1D(x, y):
    dist = np.linalg.norm(y - x)
    return dist

def Find_same(x, Q ,index_num):
    """
    :param x:一个样本
    :param Q: 一串样本
    :return: 一串样本中与一个样本欧式距离最近的样本
    """
    distance = []
    for i in range(len(Q)):
        distance.append(E_distance_1D(x, Q[i, :]))
    min_index = np.argmin(distance, axis=0)

    while min_index in index_num:
        distance[min_index] = 1e100
        min_index = np.argmin(distance,axis=0)
    index_num.append(min_index)
    return Q[min_index, :]

def CLUSTER(Xs, Xt):  # 聚类函数
    """
    :param Xs: 源域数目，即需要被自适应调整的数据集
    :param Xt: 种子数据，这里选择的可以是目标域数据
    :return: 返回聚类之后的源域数据集
    """
    # 这里求得的聚类中心，顺序是标签的顺序，也就是第一个中心对应标签值为0时的类别；
    index_num = []
    number = np.shape(Xt)[0]
    estimator = KMeans(n_clusters=number, init='k-means++')
    Xs = Xs.cpu().detach().numpy()
    estimator.fit(Xs)
    label_pred = estimator.labels_  # 获取聚类标签,这个聚类标签是非常重要的，这个后续想要吧输出也统一起来也需要用这个！
    centroids = estimator.cluster_centers_  # 获取聚类中心
    numSamples = len(Xs)
    new_Xt = []
    # 这里得到的new_Xt内涵是这样的：他表示和第一个聚类中心最接近的目标域样本点；
    for find in range(len(Xt)):  # 这个是用来将聚类中心和目标域的样本匹配上来：
        new_Xt.append(Find_same(centroids[find, :], Xt.cpu().detach().numpy(),index_num))
    new_Xt = np.array(new_Xt)
    new_Xs = copy.deepcopy(Xs)  # 这里使用的是变量的深拷贝
    Residual = np.zeros((np.shape(new_Xs)[0],np.shape(new_Xs)[1]))
    for i in range(numSamples):
        distance = E_distance_2D(centroids[label_pred[i]], Xs[i, :])  # 构成一个2D的距离矩阵，这里的distance是1*64的数据
        new_Xs[i, :] = new_Xt[label_pred[i], :] + distance
        Residual[i,:] = distance
    '''
    tsne3d = Em.TSNE(n_components=3, init='pca', random_state=0)
    X_tsne_3d = tsne3d.fit_transform(new_Xs)
    Em.plot_embedding_3d(X_tsne_3d[:, 0:3], label_pred, "new_Xs_t-SNE 3D")


    tsne3d = Em.TSNE(n_components=3, init='pca', random_state=0)
    X_tsne_3d = tsne3d.fit_transform(Xs)
    Em.plot_embedding_3d(X_tsne_3d[:, 0:3], label_pred, "Xs_t-SNE 3D")
    '''
    '''
    tsne2d = Em.TSNE(n_components=2, init='pca', random_state=0)
    X_tsne_2d = tsne2d.fit_transform(new_Xs)
    Em.plot_embedding_2d(X_tsne_2d[:, 0:2], label_pred, "new_Xs_t-SNE 2D")

    tsne2d = Em.TSNE(n_components=2, init='pca', random_state=0)
    X_tsne_2d = tsne2d.fit_transform(Xs)
    Em.plot_embedding_2d(X_tsne_2d[:, 0:2], label_pred, "Xs_t-SNE 2D")
    '''
    return new_Xs, label_pred, Residual