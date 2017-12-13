#This code has been developed as part of the class project for predicting the soil moisture content using LSTM
#CIS 731 Artificial Neural Networks
#code that has helped to develop this code is from https://machinelearningmastery.com/multivariate-time-series-forecasting-lstms-keras/

from math import sqrt
from numpy import concatenate
from pandas import read_csv
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
import numpy as np
import matplotlib.pyplot as plt

i=1
#array to hold the root mean square error values for each epoch
rmse_50 = []
index = []
num_epochs = 30
while i<=num_epochs:
    #read data from csv and convert the feature values into float
    dataset = read_csv('674_exponential_ma.csv', header=0, index_col=0)
    values = dataset.values
    values = values.astype('float32')
    #encode data values and scale or normalize them with in the range of 0 to 1
    encoder = LabelEncoder()
    #values[:, 4] = encoder.fit_transform(values[:, 4])
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled = scaler.fit_transform(values)
    values = scaled

    #four years of data total data = 4 years  for training and 7 months for testing
    n_train_hours = 35064 #(len(values)/((366+(3*365))*24))*30000
    #print(n_train_hours)
    train = values[:n_train_hours, :]
    #predicting one month soil moisture during testing
    test = values[n_train_hours:n_train_hours+(24*30*3), :]
    train_X, train_y = train[:, 1:], train[:, 0]
    test_X, test_y = test[:, 1:], test[:, 0]
    train_X = train_X.reshape((train_X.shape[0], 1, train_X.shape[1]))
    test_X = test_X.reshape((test_X.shape[0], 1, test_X.shape[1]))

    #LSTM model
    model = Sequential()
    model.add(LSTM(10, input_shape=(train_X.shape[1], train_X.shape[2])))
    model.add(Dense(1))
    model.compile(loss='mean_squared_error', optimizer='adam')
    history = model.fit(train_X, train_y, epochs=i, batch_size=72, validation_data=(test_X, test_y), verbose=2, shuffle=True)
    yhat = model.predict(test_X)
    test_X = test_X.reshape((test_X.shape[0], test_X.shape[2]))
    inv_yhat = concatenate((yhat, test_X[:, 0:]), axis=1)
    inv_yhat = scaler.inverse_transform(inv_yhat)
    inv_yhat = inv_yhat[:, 0]
    test_y = test_y.reshape((len(test_y), 1))
    inv_y = concatenate((test_y, test_X[:, 0:]), axis=1)
    inv_y = scaler.inverse_transform(inv_y)
    inv_y = inv_y[:, 0]
    inv_y_list = ['%.3f' % elem for elem in inv_y.tolist() ]
    inv_yhat_list = ['%.3f' % elem for elem in inv_yhat.tolist()]
    plot_y_list = [float(k) for k in inv_y_list]
    plot_yhat_list = [float(k) for k in inv_yhat_list]
    xrow = np.arange(len(inv_y_list))
    #plot graph at the last epoch
    if i == num_epochs:
        plt.plot(xrow,inv_yhat_list, 'r.', label='Predicted values')
        plt.plot(xrow, inv_y_list, 'b.',label='Actual values')
        plt.legend(loc='lower right', shadow=True, fontsize='x-large')
        plt.xlabel("time steps of 1 hour each")
        plt.ylabel("Value of PREC.I-1 (in)")
        plt.show()
    rmse = sqrt(mean_squared_error(inv_y, inv_yhat))
    print('Test RMSE: %.3f' % round(rmse,2))
    rmse_50.append(rmse)
    index.append(i)
    i=i+1

print(rmse_50)
#plot graph for root mean square error
plt.plot(index, rmse_50)
plt.xlabel("Number of epochs")
plt.ylabel("Root Mean Square Error")
plt.show()
print(round(sum(rmse_50),3)/num_epochs)
