import os
import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt

from sklearn.preprocessing import MinMaxScaler
from datetime import datetime

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


trainingData = nikeDataDF.iloc[:, 1:2].values #trainingData before this line is a dataframe, we need to get only open column and turn it into a numpy array to do numerical operation easier

scaler = MinMaxScaler(feature_range = (0,1)) #create a minmaxscaler and set bound
scaled_training_set = scaler.fit_transform(trainingData) #feature scaled the 2nd column