# -*- coding: utf-8 -*-
"""
Created on Wed Sep 13 09:11:30 2023

@author: Meva
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.stats as st
import importlib

# import our own files and reload
import market_data
importlib.reload(market_data)

benchmark = '^SPX' # x
security = 'GOOG' # y

# get timeseries of x and y
timeseries_x = market_data.load_timeseries(benchmark)
timeseries_y = market_data.load_timeseries(security)
timestamps_x = list(timeseries_x['date'].values)
timestamps_y = list(timeseries_y['date'].values)
timestamps = list(set(timestamps_x) & set(timestamps_y))
timeseries_x = timeseries_x[timeseries_x['date'].isin(timestamps)]
timeseries_x = timeseries_x.sort_values(by='date', ascending=True)
timeseries_x = timeseries_x.reset_index(drop=True)
timeseries_y = timeseries_y[timeseries_y['date'].isin(timestamps)]
timeseries_y = timeseries_y.sort_values(by='date', ascending=True)
timeseries_y = timeseries_y.reset_index(drop=True)
timeseries = pd.DataFrame()
timeseries['date'] = timeseries_x['date']
timeseries['close_x'] = timeseries_x['close']
timeseries['close_y'] = timeseries_y['close']
timeseries['return_x'] = timeseries_x['return']
timeseries['return_y'] = timeseries_y['return']

# plot timeseries
plt.figure(figsize=(12,5))
plt.title('Time series of close prices')
plt.xlabel('Time')
plt.ylabel('Prices')
ax = plt.gca()
ax1 = timeseries.plot(kind='line', x='date', y='close_x', ax=ax, grid=True,\
                          color='blue', label=benchmark)
ax2 = timeseries.plot(kind='line', x='date', y='close_y', ax=ax, grid=True,\
                          color='red', secondary_y=True, label=security)
ax1.legend(loc=2)
ax2.legend(loc=1)
plt.show()

# compute linear regression
x = timeseries['return_x'].values
y = timeseries['return_y'].values
slope, intercept, r_value, p_value, std_err = st.linregress(x,y)
nb_decimals = 4
alpha = np.round(intercept, nb_decimals)
beta = np.round(slope, nb_decimals)
p_value = np.round(p_value, nb_decimals) 
null_hypothesis = p_value > 0.05 # p_value < 0.05 --> reject null hypothesis
correlation = np.round(r_value, nb_decimals) # correlation coefficient
r_squared = np.round(r_value**2, nb_decimals) # pct of variance of y explained by x
predictor_linreg = intercept + slope*x

# plot linear regression
x = timeseries['return_x'].values
y = timeseries['return_y'].values
str_self = 'Linear regression | security ' + security\
    + ' | benchmark ' + benchmark + '\n'\
    + 'alpha (intercept) ' + str(alpha)\
    + ' | beta (slope) ' + str(beta) + '\n'\
    + 'p-value ' + str(p_value)\
    + ' | null hypothesis ' + str(null_hypothesis) + '\n'\
    + 'correl (r-value) ' + str(correlation)\
    + ' | r-squared ' + str(r_squared)
str_title = 'Scatterplot of returns' + '\n' + str_self
plt.figure()
plt.title(str_title)
plt.scatter(x,y)
plt.plot(x, predictor_linreg, color='green')
plt.ylabel(security)
plt.xlabel(benchmark)
plt.grid()
plt.show()
