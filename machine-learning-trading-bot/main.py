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
dataStartDate = todayDate - relativedelta(years=3)

if os.path.exists(stockDataFileName):
    stockData = pd.read_csv(stockDataFileName, index_col=0, parse_dates=True)
else:
    try:
        stockData = yf.download(ticker, start=dataStartDate, end=todayDate)
        stockData.to_csv(stockDataFileName, index=True)
    except Exception as e:
        print(f"An error occured fetching stock data: {e}")


stockPricesOnly = stockData.loc[:, ['Close']]
print(stockPricesOnly)
