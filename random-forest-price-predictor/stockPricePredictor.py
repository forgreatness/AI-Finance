import pandas as pd
import yfinance as yf
import os
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import precision_score

###############################################        Project Initialization           #########################################
"""
    - Loading the right libraries
    - Loading the data from csv file or some where
"""

###############################################         Phase 1: Data Preprocessing             ##############################################
"""
    - Clean the data
    - Normalize the data
    - feature engineer the data
    - Divide the data into features selection and y-output
"""

if os.path.exists('sp500.csv'):
    sp500 = pd.read_csv('sp500.csv', index_col=0)
else:
    sp500 = yf.Ticker("^GSPC") #represent the ticker for sp500
    sp500 = sp500.history(period="max")
    sp500.to_csv('sp500.csv')

# print(sp500)
# print(sp500.index) #returns a series of row index

# index allow to slice dataframe easily

# sp500.index = pd.to_datetime(sp500.index)
sp500.plot.line(y="Close", use_index=True) 

del sp500["Dividends"] # notice how the bracket allows use ot delete the entire column with just the bracket and name of column
del sp500["Stock Splits"] 


"""
# To work with data, we need to know exactly what it looks like first: here are ways to get the columns of the df

"""
# we did the practice for the prompt above in the dataframe-playground practice.py

# Feature engineering creating a colun call Tomorow with the idea that we need to predict this price
sp500['Tomorrow'] = sp500['Close'].shift(-1) # this lines makes a new column in our dataframe call tomorrow and assign as value closing column shift back by 1 index. Push all closing price back by 1 index
sp500['Target'] = (sp500["Tomorrow"] > sp500["Close"]).astype(int) # this one create a new column target and assign it an int of boolean value 0|1 if tommorow price is > closing

sp500 = sp500.loc["1990-01-01":].copy() #only take rows where row index which is the date is from 1990-01-01 (this is call cleaning data kinda like removing rows with null value)


############################################           Phase 2: Model Selection               #####################################################
"""
We will try Decision Tree / Random Forest
"""

model = RandomForestClassifier(n_estimators=100, min_samples_split=100, random_state=1)
train = sp500.iloc[:-100]
test = sp500.iloc[-100:]

predictors = ["Close", "Volume", "Open", "High", "Low"]
model.fit(train[predictors], train["Target"])
preds = model.predict(test[predictors])

preds = pd.Series(preds, index=test.index)
score = precision_score(test["Target"], preds)

combined = pd.concat([test["Target"], preds], axis=1)
combined.plot()
# plt.show()



def predict(train, test, predictors, model):
    model.fit(train[predictors], train["Target"])
    preds = model.predict(test[predictors]) #this always return just a list of predictions
    preds = pd.Series(preds, index=test.index, name="Predictions") #this converts the list into pandas Series where the label of each row is the datetime coming from test.index and we name the series column predictions
    combined = pd.concat([test["Target"], preds], axis=1) #this concat a new column with the pandas series into a dataframe its now 2 columns with the target column as well
    return combined

def backtest(data, model, predictors, start=2500, step=250): #dont take the first 10 year of data it seem
    all_predictions = []
    
    for i in range (start, data.shape[0], step): #start at 2500 every time we run loop increment 250 data.shape[0] how many rows of data
        train = data.iloc[0:i].copy() #0 to 2500 first then 0-2750, and then 0-3000 and etc...
        test = data.iloc[i:(i+step)].copy() #2500-2750 first the 2750 to 3000
        predictions = predict(train, test, predictors, model) # call predict function which will train model based on passed in data
        all_predictions.append(predictions)
        
    return pd.concat(all_predictions)

predictions = backtest(sp500, model, predictors)
print("I want to see all my backtest predictions", predictions)
print(predictions['Predictions'].value_counts())
    
precision_score(predictions['Target'], predictions['Predictions'])
predictions['Target'].value_counts() / predictions.shape[0]




############################################           Phase 3: Fine Tuning                   #######################################################
horizons = [2,5,60,250,1000] #num representing day to calculate average the last 2, 5, 60, ....
newPredictors = []

for horizon in horizons:
    rollingAverages = sp500.rolling(horizon).mean()
    
    ratio_column = f"Close_Ratio_{horizon}"
    sp500[ratio_column] = sp500["Close"] / rollingAverages["Close"]
    
    trend_column = f"Trend_{horizon}"
    sp500[trend_column] = sp500.shift(1).rolling(horizon).sum()["Target"] #Create a new column call trend_5, trend_60 and calculate the window of 5 or 60 days 
    
    newPredictors += [ratio_column, trend_column]
    
sp500 = sp500.dropna()

model = RandomForestClassifier(n_estimators=200, min_samples_split=50, random_state=1)

def predict2(train, test, predictors, model):
    model.fit(train[predictors], train['Target'])
    preds = model.predict_proba(test[predictors])[:,1]
    preds[preds >= .6] = 1
    preds[preds < .6] = 0
    preds = pd.Series(preds, index=test.index, name="Predictions")
    combined = pd.concat([test['Target'], preds], axis=1)
    return combined


predictions = backtest(sp500, model, newPredictors)
print(predictions['Predictions'].value_counts())

############################################            Phase 4: Model Evaluation             ######################################################