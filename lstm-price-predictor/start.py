"""
We are going to be practicing creating an LSTM to predict the stock price of APPLE.
LSTM is known for its ability to work with sequence data because it can keep track of its memory

"""

import math
import yfinance as yf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt #this library we know is using to plot data for visualization (matplotlib)
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense, LSTM
plt.style.use('fivethirtyeight')

try:
    # Fetch the data using yfinance
    df = yf.download('AAPL', start='2012-01-01', end='2019-12-17')
    print(df)
except Exception as e:
    print(f"An error occurred: {e}")
    
    
print(df.shape)

# Visualize the closing price (yeah this takes the closing price column from the df['Close'])
plt.figure(figsize=(16,8))
plt.title('Close Price History')
plt.plot(df['Close'])
plt.xlabel('Date', fontsize=18)
plt.ylabel('Close Price USD ($)', fontsize=18)
# plt.show()


data = df.filter(['Close']) 
dataset = data.values #this gets all the closing price value in dataset
training_data_len = math.ceil(len(dataset) * .8) #this convert all dataset only use 80%

# Preprocessing
# Scale the data
scaler = MinMaxScaler(feature_range=(0,1)) #scale all closing price to 0-1
scaled_data = scaler.fit_transform(dataset) 

#Create training data set
#Create teh scaled training data set
train_data = scaled_data[0:training_data_len, :] #0 - 80% of dataset as training data (scaled_data here is still a dataframe)
x_train = []
y_train = []

#Split the data into x_train and y_train data sets (the purpose of the below is to put 60 closing price per 1 prediction price)
for i in range(60, len(train_data)):
    x_train.append(train_data[i-60:i, 0]) #x_train now becomes an 1d array with elements that are 60 values array inside
    y_train.append(train_data[i, 0])
        
x_train, y_train = np.array(x_train), np.array(y_train) # this makes it so that x_train is now a 2d array

#Reshape the data (number of cases, 60 time steps, and 1 feature (closing price))
x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1)) #this converts 2d array into 3d array (which is 80% of rows from dataset, each row is 60 unit, and third dimension is 1)

####################### STEP #2 Model Creation ###############################
model = Sequential()
model.add(LSTM(50, return_sequences=True, input_shape = (x_train.shape[1], 1)))
model.add(LSTM(50, return_sequences=False))
model.add(Dense(25))
model.add(Dense(1))

# model.compile(optimizer='adam', loss='mean_squared_error')
# model.fit(x_train, y_train, batch_size=1, epochs=1)
