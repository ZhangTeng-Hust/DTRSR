# -*- coding: utf-8 -*-
"""
Created on 14:11,2021/09/13
@author: ZhangTeng
"""
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

def predict(y_true,y_pred):
    '''
    y_true: 实际标签
    y_pred: 预测标签
    return: 返回四个评价指标所组成的array
    '''

    MAE = mean_absolute_error(y_true,y_pred)
    MAPE = np.mean(np.abs((y_true - y_pred) / y_true))
    RMSE = np.sqrt(mean_squared_error(y_true,y_pred))
    R2 = r2_score(y_true,y_pred)
    print("MAE  {}".format(mean_absolute_error(y_true,y_pred)))
    print("MAPE {}".format(np.mean(np.abs((y_true - y_pred) / y_true))))
    print("RMSE  {}".format(np.sqrt(mean_squared_error(y_true,y_pred))))
    print("R2   {}".format(r2_score(y_true,y_pred)))
    result = np.array([MAE,MAPE,RMSE,R2])
    return result
