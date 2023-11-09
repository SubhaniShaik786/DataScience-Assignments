# -*- coding: utf-8 -*-
"""
Created on Fri Nov  3 10:49:35 2023

@author: sksha
"""
# Load the data
import pandas as pd
import numpy as np
df = pd.read_excel('CocaCola_Sales_Rawdata.xlsx',index_col='Quarter')
df.info()
df.shape
df.head()

# Simple Exponential Smoothing (SES):
# Fit the SES model
from statsmodels.tsa.holtwinters import SimpleExpSmoothing
model_ses = SimpleExpSmoothing(df['Sales']).fit()

# Generate forecasts for the same period as the observed data
forecast_ses = model_ses.fittedvalues
forecast_ses
# Calculate RMSE
from sklearn.metrics import mean_squared_error
from math import sqrt
rmse_ses = sqrt(mean_squared_error(df['Sales'], model_ses.fittedvalues))
rmse_ses  #406.01441084022633


# Holt-Winters Exponential Smoothing (HWES):
# Fit the HWES model
from statsmodels.tsa.holtwinters import ExponentialSmoothing
model_hwes = ExponentialSmoothing(df['Sales'], seasonal='add', seasonal_periods=4).fit()

# Calculate RMSE
from sklearn.metrics import mean_squared_error
from math import sqrt
rmse_hwes = sqrt(mean_squared_error(df['Sales'], model_hwes.fittedvalues))
rmse_hwes    #196.80794074352144

# Autoregressive Integrated Moving Average (ARIMA):
# Fit the ARIMA model
from statsmodels.tsa.arima.model import ARIMA
model_arima = ARIMA(df['Sales'], order=(1, 1, 1)).fit()

forecast_arima = model_arima.fittedvalues

# Calculate RMSE
rmse_arima = sqrt(mean_squared_error(df['Sales'][1:], forecast_arima))

# Note: ARIMA requires differencing, so we use df['Sales'][1:] to exclude the first value.

