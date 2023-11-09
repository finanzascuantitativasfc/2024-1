# -*- coding: utf-8 -*-
"""
Created on Tue Nov  7 12:20:48 2023

@author: Meva
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.stats as st
import importlib
import random
import scipy.optimize as op

# import our own files and reload
import market_data
importlib.reload(market_data)
import capm
importlib.reload(capm)
import portfolio
importlib.reload(portfolio)

# inputs
notional = 15 # in mn USD
universe = ['^SPX','^IXIC','^MXX','^STOXX','^GDAXI','^FCHI','^VIX',\
            'XLK','XLF','XLV','XLE','XLC','XLY','XLP','XLI','XLB','XLRE','XLU',\
            'SPY','EWW',\
            'IVW','IVE','QUAL','MTUM','SIZE','USMV',\
            'AAPL','MSFT','NVDA','AMZN','GOOG','META','NFLX',\
            'BRK-B','JPM','V','MA','BAC','MS','GS','BLK',\
            'LLY','JNJ','PG','MRK','ABBV','PFE',\
            'BTC-USD','ETH-USD','SOL-USD','USDC-USD','USDT-USD','DAI-USD',\
            'EURUSD=X','GBPUSD=X','CHFUSD=X','SEKUSD=X','NOKUSD=X','JPYUSD=X','MXNUSD=X']
rics = random.sample(universe, 10)

# rics = ['^SPX','^IXIC','^MXX','^STOXX','^GDAXI']
# rics = ['XLK','XLF','XLV','XLE','XLC','XLY','XLP','XLI','XLB','XLRE','XLU']
# rics = ['IVW','IVE','QUAL','MTUM','SIZE','USMV']
# rics = ['AAPL','MSFT','NVDA','AMZN','GOOG','META','NFLX']
# rics = ['BRK-B','JPM','V','MA','BAC','MS','GS','BLK']
# rics = ['LLY','JNJ','PG','MRK','PFE']
# rics = ['BTC-USD','ETH-USD','SOL-USD','USDC-USD','USDT-USD','DAI-USD']
# rics = ['BTC-USD','ETH-USD','SOL-USD']
# rics = ['USDC-USD','USDT-USD','DAI-USD']
# rics = ['AAPL','MSFT','NVDA','AMZN','GOOG','META','NFLX',\
#         'BRK-B','JPM','V','MA','BAC','MS','GS','BLK',\
#         'LLY','JNJ','PG','MRK','ABBV','PFE']
rics = ['XLC', 'XLU', 'QUAL', 'BTC-USD', 'XLY', \
        'NOKUSD=X', 'XLB', '^GDAXI', 'GOOG', 'CHFUSD=X']
# rics = universe

# efficient frontier
target_return = 0.075
include_min_variance = True
dict_portfolios = portfolio.compute_efficient_frontier(rics, notional, target_return, include_min_variance)
print(rics)
dict_portfolios['markowitz-target'].plot_histogram()
