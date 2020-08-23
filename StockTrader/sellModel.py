import tensorflow as tf
from tensorflow.keras import datasets, layers, models
import pandas as pd
import numpy as np
import matplotlib as plt
from sklearn.model_selection import train_test_split

def processData(fileName):
    df = pd.read_csv(fileName)
    df['G/B Sell'] = 1 or 0 ## insert calculation to decide
    ## insert more changes
    return df


def buildSellModel(trainData, testData, trainLabels, testLabels):
    model = models.Sequential([ ## edit to fit needs. need to change input shape
        layers.Dense(64, activation='relu', input_shape=len(testData.keys())),
        layers.Dense(64, activation='relu'),
        layers.Dropout(0.2),
        layers.Dense(128, activation='relu'),
        layers.Dropout(0.2),
        layers.Dense(1, activation='softmax') ## possible is layers.Softmax(dtype='float64')
    ])
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metric=['accuracy'])
    model.fit(trainData, trainLabels, epochs=30)
    model.evaluate(testData, testLabels)
    return model


if __name__ == "__main__":
    positions = ['TSLA', 'GOOG', 'AMZN', 'AAPL']
    filename = 'GenZ'
    data = processData(filename)
    train_data = data.sample(frac=0.8, random_state=0)
    test_data = data.drop(train_data.index)
    train_label = train_data.pop('G/B Sell')    ## if this does not work surround in []
    test_label = test_data.pop('G/B Sell')
    sellmodel = buildSellModel(train_data, test_data, train_label, test_label)
    for i in positions:
        output = sellmodel.predict[## add all current info here]
        if output > 0.95:
            print('Sell' + i)
            print(output + 'confidence')
