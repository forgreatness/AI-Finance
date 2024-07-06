import os
import pandas as pd
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
from sklearn import svm
from sklearn.preprocessing import StandardScaler
from pandas.tseries.offsets import DateOffset
from sklearn.metrics import classification_report
import yfinance as yf
from datetime import datetime
from dateutil.relativedelta import relativedelta

# since we like to use most recent data, we will use yfinance to download the data for the stock we are interest in the most
# that stock is JPM

stockDataFileName = 'jpm-stock-data.csv'
ticker = 'jpm'
todayDate = datetime.now()
dataStartDate = todayDate - relativedelta(years=10)

if os.path.exists(stockDataFileName):
    stockData = pd.read_csv(stockDataFileName, index_col=0, parse_dates=True)
else:
    try:
        stockData = yf.download(ticker, start=dataStartDate, end=todayDate)
        stockData.to_csv(stockDataFileName, index=True)
    except Exception as e:
        print(f"An error occured fetching stock data: {e}")


processedStockData = stockData.loc[:, ['Close']]
processedStockData['Actual Returns'] = processedStockData['Close'].pct_change()
processedStockData = processedStockData.dropna()

# Generate SMA trading signal
short_window = 50
long_window = 200

processedStockData['SMA_short'] = processedStockData['Close'].rolling(window=short_window).mean()
processedStockData['SMA_long'] = processedStockData['Close'].rolling(window=long_window).mean()
processedStockData = processedStockData.dropna()

# initialize signal for each stock price instance
processedStockData['Signal'] = 0.0
processedStockData.loc[(processedStockData['Actual Returns'] >= 0), 'Signal'] = 1
processedStockData.loc[(processedStockData['Actual Returns'] < 0), 'Signal'] = -1

# Actual Returns is whether today price was higher than yesterday in percentage (+/-)
# Signal is whether today price was higher than yesterday in (true false (1/0))
# shifting means we are looking at actual returns of yesterday compare to day before, it its up then the strategy return is 
# essential today return times signal from yesterday
processedStockData['Strategy Returns'] = processedStockData['Actual Returns'] * processedStockData['Signal'].shift()

# I need to plot the Strategy Returns column so we can see the new column we created
plt.figure(figsize=(10, 6))
# Take the strategy returns which are yesterday buy or not 
plt.plot(processedStockData.index, (1 + processedStockData['Strategy Returns']).cumprod(), label='Strategy Returns', color='blue')
plt.xlabel('Date')
plt.ylabel('Strategy Returns')
plt.title('Strategy Returns Over Time')

plt.legend()
plt.grid(True)
# plt.show()


x = processedStockData[['SMA_short', 'SMA_long']].shift().dropna() #takes the sma_short and sma_long and put it in a new dataframe
y = processedStockData['Signal'] #actual values are the signal which means if today close price compare to yesterday is greater or less

training_begin = x.index.min()
training_end = x.index.min() + DateOffset(months=6)

x_train = x.loc[training_begin:training_end]
y_train = y.loc[training_begin:training_end]
x_test = x.loc[training_end+DateOffset(hours=1):]
y_test = y.loc[training_end+DateOffset(hours=1):]

scaler = StandardScaler()
x_scaler = scaler.fit(x_train)
x_train_scaled = x_scaler.transform(x_train)
x_test_scaled = x_scaler.transform(x_test)

# Choose the model to train and fit data
svmModel = svm.SVC()
svmModel = svmModel.fit(x_train_scaled, y_train)
svmPred = svmModel.predict(x_test_scaled)

svm_testing_report = classification_report(y_test, svmPred)
print("Below is the testing report for the SVM model: \n", svm_testing_report)

predictionsDF = pd.DataFrame(index=x_test.index)
predictionsDF['Predicted'] = svmPred
predictionsDF['Actual Returns'] = processedStockData['Actual Returns']
predictionsDF['Strategy Returns'] = processedStockData['Actual Returns'] * predictionsDF['Predicted']

print(predictionsDF)

plt.plot(predictionsDF.index, (1 + predictionsDF[['Actual Returns', 'Strategy Returns']]).cumprod())
plt.show()



