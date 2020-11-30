
# import all the necessary libraries
import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt
# %matplotlib inline
from pandas import datetime
import math, time
import itertools
from sklearn import preprocessing
import datetime
from operator import itemgetter
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from math import sqrt
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation
from keras.layers.recurrent import LSTM
from keras.models import load_model
import keras
import h5py
import requests
import os

#import datasets
data =  pd.read_csv("prices-split-adjusted.csv", index_col = 0)
data.head()

# preprocess the data
data = data[data.symbol == 'AAPL']
data.drop(['symbol'],1,inplace=True)

# plot the target 
plt.plot(data['close'])
plt.show()


data['date'] = data.index


data['date'] = pd.to_datetime(data['date'])


# feature scaling the data using minmax
min_max_scaler = preprocessing.MinMaxScaler(feature_range=(0, 1))
dataset = min_max_scaler.fit_transform(data['close'].values.reshape(-1, 1))


# divide train and and test data
train_samples_num = int(len(dataset) * 0.7)
train_data, test_data = dataset[0:train_samples_num,:], dataset[train_samples_num:len(dataset),:]
# print(len(train), len(test))

# prepare the dataset instandard way without using any library
# with look back 15 as it can keep previous 15 states
def prepare_dataset(dataset, look_back=15):
    data_x, data_y = [], []
    for i in range(len(dataset)-look_back-1):
        a = dataset[i:(i+look_back), 0]
        data_x.append(a)
        data_y.append(dataset[i + look_back, 0])
    return np.array(data_x), np.array(data_y)


x_train, y_train = prepare_dataset(train_data, look_back=15)
x_test, y_test = prepare_dataset(test_data, look_back=15)


# print(x_train.shape)
# print(y_train.shape)
# print(x_test.shape)
# print(y_test.shape)

# reshape the features
x_train = np.reshape(x_train, (x_train.shape[0], 1, x_train.shape[1]))
x_test = np.reshape(x_test, (x_test.shape[0], 1, x_test.shape[1]))


# print(x_train.shape)
# print(y_train.shape)
# print(x_test.shape)
# print(y_test.shape)


# define simple model
look_back = 15
model = Sequential()
model.add(LSTM(20, input_shape=(1, look_back)))
model.add(Dense(1))
model.compile(loss='mean_squared_error', optimizer='adam')
model.fit(x_train, y_train, epochs=20, batch_size=1, verbose=2)


train_predict = model.predict(x_train)
predict = model.predict(x_test)
# invert predictions
train_predict = min_max_scaler.inverse_transform(train_predict)
# trainY = min_max_scaler.inverse_transform([y_train])
predict = min_max_scaler.inverse_transform(predict)
testY = min_max_scaler.inverse_transform([y_test])
# calculate root mean squared error
# trainScore = math.sqrt(mean_squared_error(trainY[0], trainPredict[:,0]))
# print('Train Score: %.2f RMSE' % (trainScore))
testScore = math.sqrt(mean_squared_error(testY[0], predict[:,0]))
print('Test Score: %.2f RMSE' % (testScore))


# shift train predictions for plotting
# trainPredictPlot = np.empty_like(dataset)
# trainPredictPlot[:, :] = np.nan
# trainPredictPlot[look_back:len(trainPredict)+look_back, :] = trainPredict
# shift test predictions for plotting
pred_plot = np.empty_like(dataset)
pred_plot[:, :] = np.nan
pred_plot[len(train_predict)+(look_back*2)+1:len(dataset)-1, :] = predict
# plot baseline and predictions
plt.plot(min_max_scaler.inverse_transform(dataset))
# plt.plot(trainPredictPlot)
plt.plot(pred_plot)
plt.show()