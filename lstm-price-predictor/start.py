"""
We are going to be practicing creating an LSTM to predict the stock price of APPLE.
LSTM is known for its ability to work with sequence data because it can keep track of its memory

"""

import math
import yfinance as yf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt #this library we know is using to plot data for visualization (matplotlib)
from datetime import datetime
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense, LSTM
plt.style.use('fivethirtyeight')

todayDate = datetime.now().strftime('%Y-%m-%d')
trainingStartDate = f'{datetime.now().year - 12}-01-01'
trainingEndDate = f'{datetime.now().year}-01-01'

try:
    # Fetch the data using yfinance
    df = yf.download('AAPL', start=trainingStartDate, end=trainingEndDate)
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

model.compile(optimizer='adam', loss='mean_squared_error')
model.fit(x_train, y_train, batch_size=1, epochs=1)

test_data = scaled_data[training_data_len - 60:, :] # takes the scaled data which is closing price column with feature scaled using normalization (80% - 60) to the end

x_test = [] #for each datapoint in y_test, we need previous 60 price from that point we do this separately like above
y_test = dataset[training_data_len:, :] #from 80% of originial data set until the end makes sense (this takes the data from dataset which is not scaled and takes it straight from 80% to the end)


for i in range (60, len(test_data)): #60 to the end of test_data meaning the first 60 of test_data is not really looped through
    x_test.append(test_data[i-60:i, 0]) #append the last 60 value before i to be the value in x_test == [i-60:i, 0] 0 means first column

x_test = np.array(x_test)
x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))

predictions = model.predict(x_test)
predictions = scaler.inverse_transform(predictions)

# Compute RMSE for model evaluation (usually RMSE is a loss function)
rmse = np.sqrt( np.mean( predictions - y_test )**2 )

print('this is the RMSE: ', rmse)


#Plot the results we have which is the ypreds comparing to the y_test
train = data[:training_data_len]
valid = data[training_data_len:]
valid['Predictions'] = predictions
#visualize these data
plt.figure(figsize=(16,8))
plt.title('Model')
plt.xlabel('Date', fontsize=18)
plt.ylabel('Close Price USD ($)', fontsize=18)
plt.plot(train['Close'])
plt.plot(valid[['Close', 'Predictions']])
plt.legend(['Train', 'Val', 'Predictions'], loc='lower right')
# plt.show()

##############################  Get price for Apple the last 60 days input into model to get prediction and get the quote price for apple tomorrow and compare actual quote price vs model pred ###################
appleEndDate = datetime.now().strftime('%Y-%m-%d')
appleStartDate = f'{datetime.now().year}-01-01'
appleData = yf.download('AAPL', start=appleStartDate, end=appleEndDate)

appleClose = appleData.filter(['Close'])
appleClose = appleClose[-60:].values #.values get us an array
appleCloseScaled = scaler.transform(appleClose)
appleXTest = []
appleXTest.append(appleCloseScaled)
appleXTest = np.array(appleXTest)
appleXTest = np.reshape(appleXTest, (appleXTest.shape[0], appleXTest.shape[1], 1))
appleTomorrowPrice = model.predict(appleXTest)
appleTomorrowPrice = scaler.inverse_transform(appleTomorrowPrice)

# Download the most recent data for Apple (AAPL)
appleMostRecentData = yf.download('AAPL', period='1d', interval='1d')

# Get the closing price for the most recent day
most_recent_close_price = appleMostRecentData['Close'].iloc[0]
print('APPLE predicted price vs actual closing price', appleTomorrowPrice, most_recent_close_price)