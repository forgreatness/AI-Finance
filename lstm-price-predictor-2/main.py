import os
import numpy as np
import pandas as pd
import yfinance as yf
import math
import matplotlib.pyplot as plt

from sklearn.preprocessing import MinMaxScaler
from datetime import datetime
from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout

dataFileName = 'nikePriceData.csv'
todayDT = datetime.now().strftime('%Y-%m-%d')
nikePredictorStartDate = f'{datetime.now().year - 4}-01-01' #last year from the Jan 1st
nikePredictorEndDate = f'{datetime.now().date()}'

if os.path.exists(dataFileName):
    nikeDataDF = pd.read_csv(dataFileName)
else:
    ticker = 'NKE'
    try: 
        nikeDataDF = yf.download(ticker, start=nikePredictorStartDate, end=nikePredictorEndDate)
        nikeDataDF.to_csv(dataFileName)
    except Exception as e:
        print(f"An error occurred: {e}")

nikeDataDF.set_index('Date', inplace=True)
print(nikeDataDF)
trainingData = nikeDataDF.iloc[:, 1:2].values #trainingData before this line is a dataframe, we need to get only open column and turn it into a numpy array to do numerical operation easier

scaler = MinMaxScaler(feature_range = (0,1)) #create a minmaxscaler and set bound
scaledTrainingData = scaler.fit_transform(trainingData) #feature scaled the 2nd column
trainingDataSize = math.ceil(.8 * scaledTrainingData.size)
#-------- Lets prep x_train, y_train, x_test, y_test --------
x_train, y_train, x_test, y_test = [], [], [], []

for i in range(60, trainingDataSize):
    x_train.append(scaledTrainingData[i-60:i, 0])
    y_train.append(scaledTrainingData[i, 0])

x_train = np.array(x_train)
y_train = np.array(y_train)

# LSTM need 3 dimensional input (number of samples, number of time steps,  number of features)
x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))


regressorModel = Sequential()
regressorModel.add(LSTM(units = 50, return_sequences=True, input_shape=(x_train.shape[1], 1))) #return_sequences attribute means the LSTM layer will have an output of 
regressorModel.add(Dropout(0.2))

regressorModel.add(LSTM(units = 50, return_sequences=True))
regressorModel.add(Dropout(0.2))

regressorModel.add(LSTM(units = 50, return_sequences=True))
regressorModel.add(Dropout(0.2))

regressorModel.add(LSTM(units = 50))
regressorModel.add(Dropout(0.2))

regressorModel.add(Dense(units=1)) #the model last layer will be a densely connected means all output from previous layer will be input to this layer

regressorModel.compile(optimizer='adam', loss='mean_squared_error')
regressorModel.fit(x_train, y_train, epochs=20, batch_size=32)

testingData = scaledTrainingData[trainingDataSize-60:, :]
y_test = scaledTrainingData[trainingDataSize:, :]

for i in range(60, len(testingData)):
    x_test.append(testingData[i-60:i, 0])

x_test = np.array(x_test)
x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))

# ******** Lets predict the price of stock ******** #
predictedStockPrice = regressorModel.predict(x_test)
predictedStockPrice = scaler.inverse_transform(predictedStockPrice)
actualStockPrice = scaler.inverse_transform(y_test)

plotData = nikeDataDF.iloc[trainingDataSize:, 1:2]
plotData['predictedStockPrice'] = predictedStockPrice

plt.plot(plotData.iloc[:, 0], color='red', label='Actual Nike Stock Price') #this first plot should get actual nike stock price
plt.plot(plotData['predictedStockPrice'], color='blue', label='Predicted Nike Stock Price')
plt.title('Nike Stock Price Prediction')
plt.xlabel('Time')
plt.ylabel('Nike Stock Price')
plt.legend()
plt.show()



### TODO ####
""" in the plot data add another row next to the last one that will be the price prediction for tomorrow. You need to get index of last column, convert to date and add a day to it """