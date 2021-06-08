import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import LSTM
import math
from sklearn.metrics import mean_squared_error
import streamlit as st
import matplotlib
matplotlib.rcParams["backend"] = "TkAgg"
import plotly.express as px

# range of data to show

stock = st.text_input('stock')

displayRange = 365
epochNum = 100
batchSize = 32
# stock = 'SNAP'
tickerSymbol = stock
tickerData = yf.Ticker(tickerSymbol)
tickerDf = tickerData.history(period='max')
# displayRange = st.slider('range', min_value=100, max_value=tickerDf.shape[0])
print(tickerDf)
scaler = MinMaxScaler(feature_range=(0,1))
# @st.cache(suppress_st_warning=True)
# def mainFunction(stock, epochNum, batchSize):
tickerSymbol = stock
tickerData = yf.Ticker(tickerSymbol)
tickerDf = tickerData.history(period='max')
if (len(tickerDf) != 0):
    print(type(tickerDf))
    tickerDf.reset_index(inplace=True)
    # tickerDf = tickerDf.iloc[:, : 5]
    tickerDf[['Date', 'Close']]

    # reshape data

    df = scaler.fit_transform(np.array(tickerDf[['Close']]).reshape(-1,1))
    print(type(df))
    t = df.copy()
    # df = df[:-30, :]
    # displayRange = len(df)


    # split data
    training_size = int(len(df) * 0.65)
    test_size = len(df) - training_size
    train, test = df[0:training_size,:], df[training_size:len(df),:1]

    # convert to matrix
    def createData(data, time_step=1):
        x, y = [], []
        for i in range(len(data) - time_step - 1):
            a = data[i:(i+time_step), 0]
            x.append(a)
            y.append(data[i + time_step, 0])
        return np.array(x), np.array(y)

    # reshape data
    time_step = 100
    xTrain, yTrain = createData(train, time_step)
    xTest, yTest = createData(test, time_step)

    xTrain = xTrain.reshape(xTrain.shape[0], xTrain.shape[1], 1)
    xTest = xTest.reshape(xTest.shape[0], xTest.shape[1], 1)

    # create LSTM model
    model = Sequential()
    model.add(LSTM(50, return_sequences=True, input_shape=(100,1)))
    model.add(LSTM(50, return_sequences=True))
    model.add(LSTM(50))
    model.add(Dense(1))
    model.compile(loss='mean_squared_error', optimizer='adam')
    # model.summary()

    # fit model
    model.fit(xTrain, yTrain, validation_data=(xTest, yTest), epochs=epochNum, batch_size=batchSize, verbose=1)

    # predictions
    trainPredict = model.predict(xTrain)
    testPredict = model.predict(xTest)

    trainPredict = scaler.inverse_transform(trainPredict)
    testPredict = scaler.inverse_transform(testPredict)

    # performance
    print(math.sqrt(mean_squared_error(yTrain,trainPredict)))
    print(math.sqrt(mean_squared_error(yTest,testPredict)))

    # plot
    # look_back = 100
    # trainPlot = np.empty_like(df)
    # trainPlot[:, :] = np.nan
    # trainPlot[look_back:len(trainPredict) + look_back, :] = trainPredict
    # testPlot = np.empty_like(df)
    # testPlot[:, :] = np.nan
    # testPlot[len(trainPredict) + (look_back * 2) + 1:len(df) - 1, :] = testPredict
    # plt.plot(scaler.inverse_transform(df))
    # plt.plot(trainPlot)
    # plt.plot(testPlot)
    # plt.show()

    othertest = test[len(test) - 100: ]

    xInput = test[len(test) - 100:].reshape(1,-1)
    input = list(xInput)
    input = input[0].tolist()

    output = []
    steps = 100
    i = 0
    while i < 30:
        if (len(input) > 100):
            xInput = np.array(input[1:])
            print('{} day input {}'.format(i, xInput))
            xInput = xInput.reshape(-1, 1)
            xInput = xInput.reshape((1, steps, 1))
            # print(xInput)
            yHat = model.predict(xInput, verbose=0)
            print('{} day output {}'.format(i, yHat))
            input.extend(yHat[0].tolist())
            input = input[1:]
            output.extend(yHat.tolist())
            i = i + 1
        else:
            xInput = xInput.reshape((1, steps, 1))
            yHat = model.predict(xInput, verbose = 0)
            print(yHat[0])
            input.extend(yHat[0].tolist())
            print(len(input))
            output.extend(yHat.tolist())
            i = i + 1

            # return (output, t)

    # @st.cache(suppress_st_warning=True)
    # def graph(output, displayRange, t):
        # scaler = MinMaxScaler(feature_range=(0,1))
    day = np.arange(0, displayRange)
    predDay = np.arange(displayRange, displayRange + 30)
    # predDay = np.arange(displayRange, displayRange + 30)

    # final = df.tolist()
    # final.extend(output)
    # final = scaler.inverse_transform(final).tolist()
    # plt.plot(final)

    # plt.plot(day, scaler.inverse_transform(t[len(t) - displayRange:]))
    # plt.plot(predDay, scaler.inverse_transform(output))
    # plt.show()
    realPrice = []
    predPrice = []
    for x in scaler.inverse_transform(t[len(t) - displayRange:]):
        realPrice.append(x[0])
    for y in scaler.inverse_transform(output):
        predPrice.append(y[0])
    real = pd.DataFrame({'day': day, 'price': realPrice})
    real['type'] = 'real'
    prediction = pd.DataFrame({'day': predDay, 'price': predPrice})
    prediction['type'] = 'prediction'
    total = real.append(prediction)
    print(total)
    plot = px.line(total, x='day', y='price', color='type')
    # plot.show()

    st.plotly_chart(plot)

# data = mainFunction(stock, epochNum, batchSize)
# print(data)
# graph(data[0], displayRange, data[1])