# -*- coding: utf-8 -*-
"""
Created on Wed Oct 25 09:34:31 2023

@author: sksha
"""

# Step 1: Load and preprocess the data
import pandas as pd
import numpy as np
df = pd.read_excel("AirLines+data.xlsx")
df.info()
df.head()
df['Month'] = pd.to_datetime(df['Month'])
df.set_index('Month', inplace=True)

# Histogram and Density plot
import matplotlib.pyplot as plt
# Assuming your data is stored in a DataFrame df
plt.hist(df['Passengers'], bins=10, density=True, alpha=0.6, color='b')
df['Passengers'].plot(kind='kde', color='r')
plt.show()

# Pivot table
pivot_table = df.pivot_table(index='Month', values='Passengers', aggfunc='mean')
pivot_table


# Lag Plot
# Create a lag plot using the 'Passengers' column
pd.plotting.lag_plot(df['Passengers'])


# Scatter Plot
plt.scatter(df['Month'], df['Passengers'])
plt.xlabel('Month')
plt.ylabel('Passengers')
plt.show()

# Check for stationarity
from statsmodels.tsa.stattools import adfuller
result = adfuller(df['Passengers'])
print("ADF Statistic:", result[0])   #ADF Statistic: 1.3402479596467132
print("p-value:", result[1])         #p-value: 0.9968250481137263
if result[1] > 0.05:
    print("The data is not stationary.")
else:
    print("The data is stationary.")

# Model Selection (SARIMA)
from statsmodels.graphics.tsaplots import plot_pacf

# Compute the PACF for up to 24 lags (or another suitable number)
import matplotlib.pyplot as plt
plot_pacf(df, lags=24)
plt.show()

# Based on ACF and PACF plots, we can choose p, d, q, P, D, and Q values.

# Step 4: Model Fitting
# In this example, we'll use p=1, d=1, q=1, P=1, D=1, and Q=1.
from statsmodels.tsa.statespace.sarimax import SARIMAX
model = SARIMAX(df['Passengers'], order=(1, 1, 1), seasonal_order=(1, 1, 1, 12))
results = model.fit()

# Step 5: Model Evaluation
train_size = int(len(df) * 0.8)
train, test = df[:train_size], df[train_size:]
results = model.fit(disp=False)
start = len(train)
end = len(train) + len(test) - 1
predictions = results.predict(start=start, end=end, dynamic=False, typ='levels')

from sklearn.metrics import mean_squared_error
from math import sqrt
rmse = sqrt(mean_squared_error(test, predictions))
print("RMSE:", rmse)

# Step 6: No dummy variables are needed for SARIMA modeling.

# Step 7: Model Comparison
# SARIMA modeling has been performed with RMSE calculated.

# Forecasting future values
forecast_steps = 12  # Number of months to forecast into the future
forecast = results.get_forecast(steps=forecast_steps)
forecast_mean = forecast.predicted_mean
forecast_ci = forecast.conf_int()

# Plot the actual data, predictions, and forecasted values
import matplotlib.pyplot as plt
plt.figure(figsize=(12, 6))
plt.plot(df, label='Observed')
plt.plot(predictions, label='Predicted', color='orange')
plt.plot(forecast_mean, label='Forecasted', color='green')
plt.fill_between(forecast_ci.index, forecast_ci.iloc[:, 0], forecast_ci.iloc[:, 1], color='gray', alpha=0.2)
plt.legend()
plt.title('Airlines Passengers Forecasting')
plt.xlabel('Date')
plt.ylabel('Passengers')
plt.show()



