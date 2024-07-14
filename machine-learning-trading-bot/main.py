import os
import pandas as pd
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
from sklearn import svm
from sklearn.preprocessing import StandardScaler
from pandas.tseries.offsets import DateOffset
from sklearn.metrics import classification_report
from sklearn.neural_network import MLPClassifier
import yfinance as yf
from datetime import datetime, timedelta
from dateutil.relativedelta import relativedelta

# since we like to use most recent data, we will use yfinance to download the data for the stock we are interest in the most
# that stock is JPM

stockDataFileName = 'jpm-stock-data.csv'
ticker = 'jpm'
todayDate = datetime.now()
dataStartDate = todayDate - relativedelta(years=12)

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
processedStockData['SMA_mini'] = processedStockData['Close'].rolling(window=20).mean()
# processedStockData['SMA_jumbo'] = processedStockData['Close'].rolling(window=400).mean()
processedStockData = processedStockData.dropna()

# initialize signal for each stock price instance
processedStockData['Signal'] = 0.0
processedStockData.loc[(processedStockData['Actual Returns'] >= 0), 'Signal'] = 1
processedStockData.loc[(processedStockData['Actual Returns'] < 0), 'Signal'] = -1

print('processedStockData["Signal"].value_count()', processedStockData['Signal'].value_counts())

# Actual Returns is whether today price was higher than yesterday in percentage (+/-)
# Signal is whether today price was higher than yesterday in (true false (1/0))
# shifting means we are looking at actual returns of yesterday compare to day before, it its up then the strategy return is 
# essential today return times signal from yesterday
processedStockData['Strategy Returns'] = processedStockData['Actual Returns'] * processedStockData['Signal'].shift()

# I need to plot the Strategy Returns column so we can see the new column we created
# plt.figure(figsize=(10, 6))
# Take the strategy returns which are yesterday buy or not 
# plt.plot(processedStockData.index, (1 + processedStockData['Strategy Returns']).cumprod(), label='Strategy Returns', color='blue')
# plt.xlabel('Date')
# plt.ylabel('Strategy Returns')
# plt.title('Strategy Returns Over Time')

# plt.legend()
# plt.grid(True)
# plt.show()


x = processedStockData[['SMA_short', 'SMA_long', 'SMA_mini']].shift().dropna() #takes the sma_short and sma_long and put it in a new dataframe
y = processedStockData['Signal'] #actual values are the signal which means if today close price compare to yesterday is greater or less

training_begin = x.index.min()
training_end = x.index.min() + DateOffset(months=3)

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
yPredDF = pd.DataFrame(svmPred, columns=['predicted_values'])
print('yPredDF["predicted_values"].value_counts()', yPredDF["predicted_values"].value_counts())

svm_testing_report = classification_report(y_test, svmPred)
print("Below is the testing report for the SVM model: \n", svm_testing_report)

predictionsDF = pd.DataFrame(index=x_test.index)
predictionsDF['Predicted'] = svmPred
print('predictionDF["Predicted"].value_counts()', predictionsDF['Predicted'].value_counts())

predictionsDF['Actual Returns'] = processedStockData['Actual Returns']
predictionsDF['Strategy Returns'] = processedStockData['Actual Returns'] * predictionsDF['Predicted']

print(predictionsDF)

# Calculate the cumulative product
predictionsDF['Cumulative Actual Returns'] = (1 + predictionsDF['Actual Returns']).cumprod()
predictionsDF['Cumulative Strategy Returns'] = (1 + predictionsDF['Strategy Returns']).cumprod()

# Plot the cumulative returns
plt.figure(figsize=(10, 6))
plt.plot(predictionsDF['Cumulative Actual Returns'], label='Cumulative Actual Returns', color='blue')
plt.plot(predictionsDF['Cumulative Strategy Returns'], label='Cumulative Strategy Returns', color='green')
# predictionsDF.assign(Cumulative_Actual=(1 + predictionsDF['Actual Returns']).cumprod(),
#                     Cumulative_Strategy=(1 + predictionsDF['Strategy Returns']).cumprod()).plot(y=['Cumulative_Actual', 'Cumulative_Strategy'], figsize=(10, 6))
plt.legend()
# plt.show()



# Phase #2: Improving the model using different SMA window and different training window
"""
In phase one we train the model using sma value of 50 and sma value of 200.
Then in phase 2 we will try a different sma value base on our theories to see if the performance of the model would improve.

How did the sma 50 and the sma 200 help predict the model:
The sma50 is a value using the rollingwindow to find the mean of the last 50 closing price, this along with sma 200 is use to predict a colume call Signal (1 for up, -1 for going down)
Using the predicted value, the strategy return will be the day actual return (today closing price / yesterday closing) times * by the predicted signal. If its a negative signal and it actually predicted it.
Then the actual return is positive. since we sold before it went down.
"""

# Trying an sma that was not only 50 but now we will do sma 20 and sma400
# smallMomentumWindow = 20
# largeEconomicShiftWindow = 400

# processedStockData['SMA_mini'] = processedStockData['Close'].rolling(window=smallMomentumWindow).mean()
# processedStockData['SMA_sizemic'] = processedStockData['Close'].rolling(window=largeEconomicShiftWindow).mean()

# shifted_data = processedStockData[['SMA_mini']].shift()
# shifted_data.dropna(subset=['SMA_mini'], inplace=True)
# processedStockData[['SMA_mini']] = shifted_data
# processedStockData.dropna(inplace=True)

# print('this is the processedStockData \n', processedStockData)

# # training_begin = x.index.min()
# # training_end = x.index.min() + DateOffset(months=3)

# # x_train = x.loc[training_begin:training_end]
# # y_train = y.loc[training_begin:training_end]

# featureEngineerX = processedStockData[['SMA_short', 'SMA_long', 'SMA_mini']]
# featureEngineerY = processedStockData['Signal']
# featureEngineerTrainingBegin = featureEngineerX.index.min()
# featureEngineerTrainingEnd = featureEngineerX.index.min() + DateOffset(months=3)

# # print("featureEngineerX", featureEngineerX)

# featureEngineerX_train = featureEngineerX.loc[featureEngineerTrainingBegin:featureEngineerTrainingEnd]
# featureEngineerY_train = featureEngineerY.loc[featureEngineerTrainingBegin:featureEngineerTrainingEnd]
# featureEngineerX_test = featureEngineerX.loc[featureEngineerTrainingEnd+DateOffset(hours=1):]
# featureEngineerY_test = featureEngineerY.loc[featureEngineerTrainingEnd+DateOffset(hours=1):]

# # x_scaler = scaler.fit(featureEngineerX_train)
# # featureEngineerX_train_scaled = x_scaler.transform(featureEngineerX_train)
# # featureEngineerX_test_scaled = x_scaler.transform(featureEngineerX_test)

# featureEngineerSVMModel = svm.SVC()
# featureEngineerSVMModel = featureEngineerSVMModel.fit(featureEngineerX_train, featureEngineerY_train)
# featureEngineerYpred = featureEngineerSVMModel.predict(featureEngineerX_test) #this ypred represents signal which is up or down, we can use this signal along with returns to predict things


# featureEngineerSVMReport = classification_report(featureEngineerY_test, featureEngineerYpred)
# print("Below is the testing report for the SVM model: \n", featureEngineerSVMReport)

# featureEngineerPredictedDF = pd.DataFrame(index=featureEngineerX_test.index)
# featureEngineerPredictedDF['Predicted'] = featureEngineerYpred
# print('featureEngineerPredictedDF["Predicted"].value_counts()', featureEngineerPredictedDF['Predicted'].value_counts())

# predictionsDF['Actual Returns'] = processedStockData['Actual Returns']
# predictionsDF['Strategy Returns'] = processedStockData['Actual Returns'] * predictionsDF['Predicted']

# print(predictionsDF)

# # Calculate the cumulative product
# predictionsDF['Cumulative Actual Returns'] = (1 + predictionsDF['Actual Returns']).cumprod()
# predictionsDF['Cumulative Strategy Returns'] = (1 + predictionsDF['Strategy Returns']).cumprod()

# # Plot the cumulative returns
# plt.figure(figsize=(10, 6))
# plt.plot(predictionsDF['Cumulative Actual Returns'], label='Cumulative Actual Returns', color='blue')
# plt.plot(predictionsDF['Cumulative Strategy Returns'], label='Cumulative Strategy Returns', color='green')
# # predictionsDF.assign(Cumulative_Actual=(1 + predictionsDF['Actual Returns']).cumprod(),
# #                     Cumulative_Strategy=(1 + predictionsDF['Strategy Returns']).cumprod()).plot(y=['Cumulative_Actual', 'Cumulative_Strategy'], figsize=(10, 6))
# plt.legend()
# plt.show()



# Phase 2 | Part 2: Increasing the training window to 9 months, 11, or reduce to 4, 5
training_begin = x.index.min()
trainingEnd9Months = x.index.min() + DateOffset(months=9)
trainingEnd11Months = x.index.min() + DateOffset(months=11)
trainingEnd4Months = x.index.min() + DateOffset(months=5)
trainingEnd5Months = x.index.min() + DateOffset(months=4)

x_train4 = x.loc[training_begin:trainingEnd4Months]
x_train5 = x.loc[training_begin:trainingEnd5Months]
x_train9 = x.loc[training_begin:trainingEnd9Months]
x_train11 = x.loc[training_begin:trainingEnd11Months]
y_train4 = y.loc[training_begin:trainingEnd4Months]
y_train5 = y.loc[training_begin:trainingEnd5Months]
y_train9 = y.loc[training_begin:trainingEnd9Months]
y_train11 = y.loc[training_begin:trainingEnd11Months]

svm4 = svm.SVC()
svm5 = svm.SVC()
svm9 = svm.SVC()
svm11 = svm.SVC()
svm4.fit(x_train4, y_train4)
svm5.fit(x_train5, y_train5)
svm9.fit(x_train9, y_train9)
svm11.fit(x_train11, y_train11)

svm4YPred = svm4.predict(x_test)
svm5YPred = svm5.predict(x_test)
svm9YPred = svm9.predict(x_test)
svm11YPred = svm11.predict(x_test)

predictionsDF['Predicted4'] = svm4YPred
predictionsDF['Predicted5'] = svm5YPred
predictionsDF['Predicted9'] = svm9YPred
predictionsDF['Predicted11'] = svm11YPred

predictionsDF['Strategy Returns4'] = processedStockData['Actual Returns'] * predictionsDF['Predicted4']
predictionsDF['Strategy Returns5'] = processedStockData['Actual Returns'] * predictionsDF['Predicted5']
predictionsDF['Strategy Returns9'] = processedStockData['Actual Returns'] * predictionsDF['Predicted9']
predictionsDF['Strategy Returns11'] = processedStockData['Actual Returns'] * predictionsDF['Predicted11']

predictionsDF['Cumulative Strategy Returns 4'] = (1 + predictionsDF['Strategy Returns4']).cumprod()
predictionsDF['Cumulative Strategy Returns 5'] = (1 + predictionsDF['Strategy Returns5']).cumprod()
predictionsDF['Cumulative Strategy Returns 9'] = (1 + predictionsDF['Strategy Returns9']).cumprod()
predictionsDF['Cumulative Strategy Returns 11'] = (1 + predictionsDF['Strategy Returns11']).cumprod()

print('this is the predictionDF', predictionsDF)

plt.plot(predictionsDF['Cumulative Strategy Returns 4'], label='Cumulative Strategy Returns 4', color='pink')
plt.plot(predictionsDF['Cumulative Strategy Returns 5'], label='Cumulative Strategy Returns 5', color='yellow')
plt.plot(predictionsDF['Cumulative Strategy Returns 9'], label='Cumulative Strategy Returns 9', color='orange')
plt.plot(predictionsDF['Cumulative Strategy Returns 11'], label='Cumulative Strategy Returns 11', color='red')
plt.legend()
plt.show()

"""
# Phase 3: In this phase, the purpose will be to try different model type on the data to see the performance
"""

nnCLF = MLPClassifier(alpha=1e-5, max_iter=1000, hidden_layer_sizes=(8, 4), random_state=42)
nnCLF.fit(x_train_scaled, y_train)
nnCLFYPred = nnCLF.predict(x_test_scaled)
print(nnCLFYPred)

nnCLFTestingReport = classification_report(y_test, nnCLFYPred)
# print(nnCLFTestingReport)


# Create a predictions DataFrame
MLP_predictions_df = pd.DataFrame(index = x_test.index)

# Add the model predictions to the DataFrame
MLP_predictions_df['Predicted'] = nnCLFYPred

# Add the actual returns to the DataFrame
MLP_predictions_df['Actual Returns'] = processedStockData['Actual Returns']

# Add the strategy returns to the DataFrame
MLP_predictions_df['Strategy Returns'] = MLP_predictions_df['Actual Returns'] * MLP_predictions_df['Predicted']
MLP_predictions_df['Cumulative Strategy'] = (1 + MLP_predictions_df['Strategy Returns']).cumprod()
MLP_predictions_df['Cumulative Actual'] = (1 + MLP_predictions_df['Actual Returns']).cumprod()

# Review the DataFrame
plt.plot(MLP_predictions_df['Cumulative Actual'], label='Cumulative Actual Returns', color='orange')
plt.plot(MLP_predictions_df['Cumulative Strategy'], label='Cumulative Strategy Returns', color='blue')
plt.legend()
plt.show()