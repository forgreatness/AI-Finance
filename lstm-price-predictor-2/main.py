import os
import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt

from datetime import datetime

todayDT = datetime.now().strftime('%Y-%m-%d')
nikePredictorStartDate = f'{datetime.now().year - 1}-01-01' #last year from the Jan 1st
nikePredictorEndDate = f'{datetime.now().date()}'


