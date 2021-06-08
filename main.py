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

# st.write('hi')



tickerSymbol = 'AAL'
tickerData = yf.Ticker(tickerSymbol)

tickerDf = tickerData.history(period='max')
print(type(tickerDf))
tickerDf.reset_index(inplace=True)
# tickerDf = tickerDf.iloc[:, : 5]
tickerDf[['Date', 'Close']]
# print(tickerDf)

# plt.plot(tickerDf[['Close']])
# plt.show()

scaler = MinMaxScaler(feature_range=(0,1))
df = scaler.fit_transform(np.array(tickerDf[['Close']]).reshape(-1,1))
# print(df)

training_size = int(len(df) * 0.65)
test_size = len(df) - training_size
train, test = df[0:training_size,:], df[training_size:len(df),:1]

def createData(data, time_step=1):
    x, y = [], []
    for i in range(len(data) - time_step - 1):
        a = data[i:(i+time_step), 0]
        x.append(a)
        y.append(data[i + time_step, 0])
    return np.array(x), np.array(y)

time_step = 100
xTrain, yTrain = createData(train, time_step)
xTest, yTest = createData(test, time_step)

xTrain = xTrain.reshape(xTrain.shape[0], xTrain.shape[1], 1)
xTest = xTest.reshape(xTest.shape[0], xTest.shape[1], 1)

model = Sequential()
model.add(LSTM(50, return_sequences=True, input_shape=(100,1)))
model.add(LSTM(50, return_sequences=True))
model.add(LSTM(50))
model.add(Dense(1))
model.compile(loss='mean_squared_error', optimizer='adam')
# model.summary()

model.fit(xTrain, yTrain, validation_data=(xTest, yTest), epochs=100, batch_size=64, verbose=1)

trainPredict = model.predict(xTrain)
testPredict = model.predict(xTest)

trainPredict = scaler.inverse_transform(trainPredict)
testPredict = scaler.inverse_transform(testPredict)

math.sqrt(mean_squared_error(yTrain,trainPredict))
math.sqrt(mean_squared_error(yTest,testPredict))

look_back = 100
trainPlot = np.empty_like(df)
trainPlot[:, :] = np.nan
trainPlot[look_back:len(trainPredict) + look_back, :] = trainPredict
testPlot = np.empty_like(df)
testPlot[:, :] = np.nan
testPlot[len(trainPredict) + (look_back * 2) + 1:len(df) - 1, :] = testPredict
plt.plot(scaler.inverse_transform(df))
plt.plot(trainPlot)
plt.plot(testPlot)
plt.show()

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