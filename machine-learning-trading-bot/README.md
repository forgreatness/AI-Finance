## Purpose
The goals of this project is to create a trading bot, that will utilize machine learning to help confirm a condition for when to place a trade. The trading bot should have backtesting to see its performance, before making any real life trade. As it make trades in real life, it should have limitation place on position sizing, stop loss. 


Machine Learning Trading Bot (by ML_Trading_Bot)
Algorithm Review (Step by Step)
- Takes only the close column among all data available from dataset
- Create an actual returns column by taking the close column and compare it previous day
- Create a short window and a long window 50 and 200 create 2 more column equalling 4 with short window being taking the last 50 close value to compute mean and do same for long window of 200
- Create signal column totalling 5, this signal column is going to be 1 if actual return for that day is greater than 1
- Create another column totalling 6 called Strategy Returns by using the signal value of previous day (whether previous day was up) and times it by the actual returns for current day. If yesterday is up then yesterday time today could be negative if return is negative (positive if its positive)
plot the value of strategy returns using cumulative return to plot the return of strategy returns over time. 
- Create x and y representing feature inputs and output of the model. features will be sma_short and sma_long. output is y which is signal
- Create 2 time frame trainingbegin and trainingend 
- Filter x using the timeframe store value in x_train, filter y as well in y_train
do same for x_test and y_test


### Phase 2: Tuning baseline algorithm
For this specific section, it will display the results of feature engineering 1 additional SMA and also testing SVM model on more training data or less training data. Additionally, there are many things which can be done with data preprocessing such as data augmentation which makes data available for each class equally using smote, removing outliers

Below we will see the results of running different svm model with sma20, sma50, and sma200 on 4 different types of training window of 4 months, 5, 9, and 11 months.

We can see that 4 months, 11 months, and regular strategy returns of 6 months gave best results
![Phase 2 Results](/images/Phase2Results.png)