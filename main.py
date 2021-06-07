import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
import numpy as np

# st.write('hi')



tickerSymbol = 'AAL'
tickerData = yf.Ticker(tickerSymbol)

tickerDf = tickerData.history(period='max')
print(type(tickerDf))
tickerDf.reset_index(inplace=True)
# tickerDf = tickerDf.iloc[:, : 5]
tickerDf[['Date', 'Close']]
print(tickerDf)

# plt.plot(tickerDf[['Close']])
# plt.show()

scaler = MinMaxScaler(feature_range=(0,1))
df = scaler.fit_transform(np.array(tickerDf[['Close']]).reshape(-1,1))
print(df)

training_size = int(len(df) * 0.65)
test_size = len(df) - training_size
train, test = df[0:training_size,:], df[training_size:len(df),:1]
