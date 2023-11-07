# -*- coding: utf-8 -*-
"""
Created on Wed Oct 25 09:18:34 2023

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
rics = random.sample(universe, 5)

print(rics)

# initialise the instance of the class
port_mgr = portfolio.manager(rics, notional)

# compute correlation and variance-covariance matrix
port_mgr.compute_covariance()

# compute the desired portfolios: output class = portfolio.output
port_min_variance_l1 = port_mgr.compute_portfolio('min_variance_l1')
port_min_variance_l2 = port_mgr.compute_portfolio('min_variance_l2')
port_long_only = port_mgr.compute_portfolio('long_only')
port_equi_weight = port_mgr.compute_portfolio('equi_weight')
port_markowitz = port_mgr.compute_portfolio('markowitz', target_return=None)

# plot the histograms of returns for the desired portfolio
port_min_variance_l1.plot_histogram()
port_min_variance_l2.plot_histogram()
port_long_only.plot_histogram()
port_equi_weight.plot_histogram()
port_markowitz.plot_histogram()
    
