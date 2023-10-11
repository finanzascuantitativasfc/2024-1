# -*- coding: utf-8 -*-
"""
Created on Tue Oct 10 09:28:29 2023

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
import capm
importlib.reload(capm)

# rics = ['^MXX','^SPX','BTC-USD','MXNUSD=X',\
#         'XLK','XLF','XLV','XLP','XLY','XLE','XLI']
# rics = ['^MXX','^SPX','^IXIC', '^STOXX', '^GDAXI', '^FCHI', '^VIX',\
#         'BTC-USD', 'ETH-USD', 'SOL-USD', 'USDC-USD', 'USDT-USD', 'DAI-USD']
rics = ['AAPL','MSFT','NVDA','AMZN','GOOG','META','NFLX',\
        'BRK-B','JPM','V','MA','BAC','MS','GS','BLK',\
        'LLY','JNJ','PG','MRK','ABBV','PFE']


# synchronise all the timeseries of returns
df = market_data.synchronise_returns(rics)

# compute the variance-covariance and correlation matrices
mtx = df.drop(columns=['date'])
mtx_var_covar = np.cov(mtx, rowvar=False)
mtx_correl = np.corrcoef(mtx, rowvar=False)

# unit test for correlations
# correl = capm.compute_correlation('^SPX','BTC-USD')