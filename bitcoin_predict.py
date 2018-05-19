# -*- coding: utf-8 -*-
# @Time    : 2018/4/19 15:59
# @Author  : Ant
# @Email   : viking.ares.ant@gmail.com
# @File    : bitcoin_lstm.py
# @Version : Python 3.6

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"


from matplotlib import pyplot

## Keras for deep learning
from keras.layers.core import Dense, Activation, Dropout
from keras.layers.recurrent import LSTM
from keras.layers import Bidirectional
from keras.models import Sequential
from keras import initializers
from math import sqrt
## Scikit learn for mapping metrics
from sklearn.metrics import mean_squared_error
from sklearn import preprocessing
from sklearn.preprocessing import MinMaxScaler,StandardScaler,Normalizer

#for logging
import time

##matrix math
import numpy as np
import math

##data processing
import pandas as pd


'''
    使得 y 值的 label 也进行了归一化的操作
'''
# -----------------------------------------------------------------------
# 判断过去几天的数据来预测今天，look_back >= 1
def N_Vector(data,look_back):
    dataX,dataY = [],[]
    for i in range(len(data) - look_back):
        if i >= look_back:
            # temp = np.array(np.c_[data[i - look_back:i, 1:-3], data[i - look_back:i, -1]])
            temp = np.array(data[i - look_back:i ,1:-1])
            dataX.append(temp)
            dataY.append(data[i,-1])
    return np.array(dataX),np.array(dataY)

# 判断过去几天的数据来预测今天，look_back >= 1
def One_Vector(data,look_back):
    dataX,dataY = [],[]
    for i in range(len(data) - look_back):
        if i >= look_back:
            # a = np.array(np.c_[data[i - look_back:i, 1:-3], data[i - look_back:i, -1]])
            a = data[i - look_back:i , 1:-1]
            a = a.flatten()
            dataX.append(a)
            dataY.append(data[i,-1])
    return np.array(dataX),np.array(dataY)
# -----------------------------------------------------------------------

# N 维训练
def N_reshape_training(train_data,test_data,look_back):

    '''
        :param train_data:
        :param test_data:
        :param look_back:
        :return:  predict_Y
    '''

    '''
        数据预处理
    '''
    # ========分开处理======
    # # 缩放化
    # scaler1 = MinMaxScaler()
    # train_data = scaler1.fit_transform(train_data)
    #
    # scaler2 = MinMaxScaler()
    # test_data = scaler2.fit_transform(test_data)
    # *******--------==------========================

    # 标准化
    scaler = StandardScaler()
    scaler.fit(train_data)
    train_data = scaler.transform(train_data)
    test_data = scaler.transform(test_data)

    train_x, train_y = N_Vector(train_data, look_back)
    test_x, test_y = N_Vector(test_data, look_back)

    model = Sequential()
    model.add(LSTM(100, input_shape=(train_x.shape[1], train_x.shape[2]), activation='hard_sigmoid',
                   recurrent_activation='hard_sigmoid', use_bias=True, kernel_initializer='glorot_uniform',
                   recurrent_initializer='orthogonal', bias_initializer='zeros'))
    model.add(Dense(1))

    # model.compile(loss="mae", optimizer="adam")
    model.compile(loss='mean_squared_error', optimizer='adam')
    history = model.fit(train_x, train_y, epochs=30, batch_size=20,verbose=2,
                        shuffle=False)
    yhat = model.predict(test_x)

    # 返回 标准化的 标准值
    return (yhat * sqrt(scaler.var_[-1]) + scaler.mean_[-1]),(test_y * sqrt(scaler.var_[-1]) + scaler.mean_[-1])

    # 返回 缩放的 标准值
    # return (yhat * (scaler.data_max_[-1] - scaler.data_min_[-1]) + scaler.data_min_[-1]),(test_y * (scaler.data_max_[-1] - scaler.data_min_[-1]) + scaler.data_min_[-1])
    # return yhat,test_y

# 一维训练
def One_reshape_training(train_data,test_data,look_back):

    '''
        :param train_data:
        :param test_data:
        :param look_back:
        :return:  predict_Y
    '''

    '''
        数据预处理
    '''
    # ========分开处理======
    # # 缩放化
    # scaler1 = MinMaxScaler()
    # train_data = scaler1.fit_transform(train_data)
    #
    # scaler2 = MinMaxScaler()
    # test_data = scaler2.fit_transform(test_data)
    # *******--------==------========================

    # 标准化
    scaler = StandardScaler()
    scaler.fit(train_data)
    train_data = scaler.transform(train_data)
    test_data = scaler.transform(test_data)

    train_x,train_y = One_Vector(train_data,look_back)
    test_x,test_y = One_Vector(test_data,look_back)

    train_x = np.reshape(train_x, (train_x.shape[0], 1, train_x.shape[1]))
    test_x = np.reshape(test_x, (test_x.shape[0], 1, test_x.shape[1]))

    model = Sequential()
    model.add(LSTM(100, input_shape=(train_x.shape[1], train_x.shape[2]), activation='hard_sigmoid',
                   recurrent_activation='hard_sigmoid', use_bias=True, kernel_initializer='glorot_uniform',
                   recurrent_initializer='orthogonal', bias_initializer='zeros'))
    model.add(Dense(1))

    # model.compile(loss="mae", optimizer="adam")
    model.compile(loss='mean_squared_error', optimizer='adam')
    history = model.fit(train_x, train_y, epochs=30, batch_size=20, verbose=2,
                        shuffle=False)
    yhat = model.predict(test_x)

    return (yhat * sqrt(scaler.var_[-1]) + scaler.mean_[-1])
    # return (yhat * (scaler.data_max_[-1] - scaler.data_min_[-1]) + scaler.data_min_[-1])
    # return yhat


if __name__ == "__main__":

    num = 15
    look_back = 4

    train_data = pd.read_csv("data/train_data_" + str(num) + ".csv")
    test_data = pd.read_csv("data/test_data_" + str(num) + ".csv")

    N_y, WholeTrue = N_reshape_training(train_data,test_data,look_back)
    One_y = One_reshape_training(train_data,test_data,look_back)

    rmse_1 = sqrt(mean_squared_error(WholeTrue,N_y))
    print("N Vector RMSE:", rmse_1)
    rmse_2 = sqrt(mean_squared_error(WholeTrue,One_y))
    print("One Vector RMSE:", rmse_2)

    pyplot.plot(One_y, label="time_step_1_y")
    pyplot.plot(N_y, label="time_step_" + str(num) + "_y")
    pyplot.plot(WholeTrue, label='True')
    pyplot.legend()
    pyplot.show()

