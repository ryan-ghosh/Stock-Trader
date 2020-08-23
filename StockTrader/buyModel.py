import math
import tensorflow as tf
from tensorflow.keras import datasets, layers, models
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
import pandas_datareader as web


def processData(stock: str, date: str):
    df = web.DataReader('DAL', data_source='yahoo', start='2012-01-01', end='2020-08-17')  ## pulling data
    plt.figure(figsize=(16,8))
    plt.plot(df['Close'])
    plt.title('Close Price of MSFT')
    plt.xlabel('Date', fontsize=18)
    plt.ylabel('Close Price', fontsize=18)
    # plt.show()

    ## process data
    data = df.filter(['Close'])
    dataset = data.values

    trainDataLength = math.ceil(len(dataset) * 0.8)

    scaled_data = normalize(dataset)

    trainData = scaled_data[0:trainDataLength , :]
    
    ## separate into xtrain and ytrain
    xtrain = []
    ytrain = []
    for i in range(60, len(trainData)): # past 60 days
        xtrain.append(trainData[i-60:i,0])
        ytrain.append(trainData[i,0])

    ## convert to numpy arrays
    xtrain, ytrain = np.array(xtrain), np.array(ytrain)

    ## reshape data to 3D for LSTM model
    xtrain = np.reshape(xtrain, (xtrain.shape[0], xtrain.shape[1], 1))

    # ## create test dataset
    testData = scaled_data[trainDataLength-60:,:]

    ## create the xtest ytest
    xtest = []
    ytest = dataset[trainDataLength:,:]
    for i in range(60, len(testData)):
        xtest.append(testData[i-60:i,0])
    
    ## convert to np array
    xtest = np.array(xtest)
    xtest = np.reshape(xtest, (xtest.shape[0], xtest.shape[1], 1))

    return [xtrain, ytrain, xtest, ytest, dataset]

def build_LSTM(xtrain, ytrain):
    # build model
    model = models.Sequential([
        layers.LSTM(50, return_sequences=True, input_shape=(xtrain.shape[1],1)),   ## return sequences true since multiple lstm layers
        layers.LSTM(50, return_sequences=False),
        layers.Dense(25),
        layers.Dense(1),
    ])
    ## compiling
    model.compile(optimizer='adam', loss='mean_squared_error')

    # training
    model.fit(xtrain, ytrain, batch_size=1, epochs=1)
    # model.save('trader.h5')
    return model

def normalize(dataset):
    scaler = MinMaxScaler(feature_range=(0,1))
    scaled_data = scaler.fit_transform(dataset)
    
    return scaled_data

def getPrediction(stock: str, date: str, model, dataset):
    apple_quote = web.DataReader(stock, data_source='yahoo', start='2012-08-19', end=date)
    scaler = MinMaxScaler(feature_range=(0,1))
    scaled_data = scaler.fit_transform(dataset)
    new_df = apple_quote.filter(['Close'])
    last60days = new_df[-60:].values
    last60daysscaled = scaler.transform(last60days)
    X_Test = [last60daysscaled]
    X_Test = np.array(X_Test)
    X_Test = np.reshape(X_Test, (X_Test.shape[0], X_Test.shape[1],1))
    
    ## get predicted price
    pred_price = model.predict(X_Test)
    pred_price = scaler.inverse_transform(pred_price)
    print(last60days[-1])
    return pred_price


def getRMSE(xtest, ytest, dataset, model):
    scaler = MinMaxScaler(feature_range=(0,1))
    scaled_data = scaler.fit_transform(dataset)
    ## get predictions
    predictions = model.predict(xtest)
    predictions = scaler.inverse_transform(predictions)

    ## evaluate by getting root mean square error
    rmse = np.sqrt(np.mean(((predictions-ytest)**2)))
    return rmse


if __name__ == "__main__":
    plt.style.use('fivethirtyeight')
    df = web.DataReader('AAPL', data_source='yahoo', start='2012-01-01', end='2020-08-19')  ## pulling data
    plt.figure(figsize=(16,8))
    plt.plot(df['Close'])
    plt.title('Close Price of MSFT')
    plt.xlabel('Date', fontsize=18)
    plt.ylabel('Close Price', fontsize=18)
    # plt.show()

    ## process data
    data = df.filter(['Close'])
    dataset = data.values

    trainDataLength = math.ceil(len(dataset) * 0.8)

    ## scale data
    scaler = MinMaxScaler(feature_range=(0,1))
    scaled_data = scaler.fit_transform(dataset)
    trainData = scaled_data[0:trainDataLength , :]
    
    ## separate into xtrain and ytrain
    xtrain = []
    ytrain = []
    for i in range(60, len(trainData)): # past 60 days
        xtrain.append(trainData[i-60:i,0])
        ytrain.append(trainData[i,0])

    ## convert to numpy arrays
    xtrain, ytrain = np.array(xtrain), np.array(ytrain)

    ## reshape data to 3D for LSTM model
    xtrain = np.reshape(xtrain, (xtrain.shape[0], xtrain.shape[1], 1))
    # build model
    # model = models.Sequential([
    #     layers.LSTM(50, return_sequences=True, input_shape=(xtrain.shape[1],1)),   ## return sequences true since multiple lstm layers
    #     layers.LSTM(50, return_sequences=False),
    #     layers.Dense(25),
    #     layers.Dense(1),
    # ])
    # ## compiling
    # model.compile(optimizer='adam', loss='mean_squared_error')

    # # training
    # model.fit(xtrain, ytrain, batch_size=1, epochs=1)
    # model.save('trader.h5')

    model = models.load_model("trader.h5")
    ## create test dataset
    testData = scaled_data[trainDataLength-60:,:]

    ## create the xtest ytest
    xtest = []
    ytest = dataset[trainDataLength:,:]
    for i in range(60, len(testData)):
        xtest.append(testData[i-60:i,0])
    
    ## convert to np array
    xtest = np.array(xtest)
    xtest = np.reshape(xtest, (xtest.shape[0], xtest.shape[1], 1))

    ## get predictions
    predictions = model.predict(xtest)
    predictions = scaler.inverse_transform(predictions)

    ## evaluate by getting root mean square error
    rmse = np.sqrt(np.mean(((predictions-ytest)**2)))
    print(rmse)

    ## get new quote
    apple_quote = web.DataReader('AAPL', data_source='yahoo', start='2012-08-19', end='2020-08-20')
    new_df = apple_quote.filter(['Close'])
    last60days = new_df[-60:].values
    last60daysscaled = scaler.transform(last60days)
    X_Test = [last60daysscaled]
    X_Test = np.array(X_Test)
    X_Test = np.reshape(X_Test, (X_Test.shape[0], X_Test.shape[1],1))
    
    ## get predicted price
    pred_price = model.predict(X_Test)
    pred_price = scaler.inverse_transform(pred_price)
    print(pred_price)
    print(last60days[-1])

    # xtrain = processData('AAPL', '2020-08-19')[0]
    # ytrain = processData('AAPL', '2020-08-19')[1]
    # xtest = processData('AAPL', '2020-08-19')[2]
    # ytest = processData('AAPL', '2020-08-19')[3]
    # dataset = processData('AAPL', '2020-08-19')[4]
    
    # LSTM = models.load_model('trader.h5')
    # print(getRMSE(xtest, ytest, dataset, LSTM))
    # print(getPrediction('AAPL', '2020-08-19', LSTM, dataset))


