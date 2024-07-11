import pandas as pd
import numpy as np
import requests
import bs4 as bs
import datetime
import yfinance as yf
# import xlsxwriter
import math
import os

response = requests.get('http://en.wikipedia.org/wiki/List_of_S%26P_500_companies')
soup = bs.BeautifulSoup(response.text, 'lxml')
table = soup.find('table', {'class': 'wikitable sortable'})

tickers = []

for row in table.findAll('tr')[1:]:
    ticker = row.findAll('td')[0].text
    tickers.append(ticker)

tickers = [s.replace('\n', '') for s in tickers]

dataFileName = 'sp500StockData.csv'
if os.path.exists(dataFileName):
    data = pd.read_csv(dataFileName)
else:
    try:
        data = []
        for ticker in tickers:
            tickerObj = yf.Ticker(ticker)
            info = tickerObj.info
            marketCap = info.get('marketCap')
            stockPrice = info.get('currentPrice')
            data.append({
                'Ticker': ticker,
                'Market Capitalization': marketCap,
                'Stock Price': stockPrice,
                'Number of Shares to Buy': 'N/A'
            })
        data = pd.DataFrame(data)
        data = data.dropna()
        data.to_csv(dataFileName)
    except Exception as e:
        print(f"There was an error: {e}")
        

positionSize = 0
try:
    portfolioSize = input('Enter the value of your investing budget: ')
    budget = float(portfolioSize)
    positionSize = budget/len(data)
except Exception as e:
    print(f"Input you enter is not a number")
    
print('positionSize', positionSize)

for i in range(0, len(data)):
    data.loc[i]['Number of Shares to Buy'] = math.floor(positionSize / data.iloc[i]['Stock Price'])

print(data)