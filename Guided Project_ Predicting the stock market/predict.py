import pandas as pd
from datetime import datetime

stock = pd.read_csv('sphist.csv', parse_dates = ['Date'])

stock = stock.sort_values(by = 'Date')
print(stock.head())

stock['5 Days Open'] = stock['Open'].rolling(window = 5).mean().shift(1)
stock['5 Days High'] = stock['High'].rolling(window = 5).mean().shift(1)
stock['5 Days Low'] = stock['Low'].rolling(window= 5).mean().shift(1)
stock['5 Days Volume'] = stock['Volume'].rolling(window = 5).mean().shift(1)

print(stock.head(10))

stock = stock[stock['Date'] >= datetime(1951,1,3)]
stock.dropna(axis = 0)

print(stock.head())

train = stock[stock['Date'] < datetime(2013,1,1)]
test = stock[stock['Date'] >= datetime(2013,1,1)]


from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error
import numpy as np

features = train.columns.drop(['Close', 'High', 'Low', 'Open', 'Volume', 'Adj Close', 'Date'])


model = LinearRegression()
model.fit(train[features], train['Close'])
predict = model.predict(test[features])
mae = mean_absolute_error(test['Close'] , predict)
mse = mean_squared_error(test['Close'], predict)

print("RMSE {}".format(np.sqrt(mse)))

test_df = test.copy()
test_df['predict'] = predict
test_df = test_df[['Date', 'Close', 'predict']]
test_df['diff'] = abs(test_df['Close'] - test_df['predict'])
test_df['accuracy'] = (1 - (test_df['diff'] / test_df['Close'])) * 100


print(test_df.sort_values('Date', ascending = False))


