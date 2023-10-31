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

# initialise the instance of the class
port_mgr = portfolio.manager(rics, notional)

# compute correlation and variance-covariance matrix
port_mgr.compute_covariance()

# compute the desired portfolios: output class = portfolio.output
port_min_variance_l1 = port_mgr.compute_portfolio('min_variance_l1')
port_min_variance_l2 = port_mgr.compute_portfolio('min_variance_l2')
port_equi_weight = port_mgr.compute_portfolio('equi_weight')
port_long_only = port_mgr.compute_portfolio('long_only')
port_markowitz = port_mgr.compute_portfolio('markowitz', target_return=0.1949)

# return_target = port_long_only.target_return
return_portfolio_long_only = np.round(port_mgr.returns.dot(port_long_only.weights), 6)
return_portfolio_equi_weight = np.round(port_mgr.returns.dot(port_equi_weight.weights), 6)
return_portfolio_markowitz = np.round(port_mgr.returns.dot(port_markowitz.weights), 6)


# df = pd.DataFrame()
# df['rics'] = rics
# df['returns'] = port_mgr.returns 
# df['volatilities'] = port_mgr.volatilities 
# df['markowitz_weights'] = port_markowitz.weights 
# df['markowitz_allocation'] = port_markowitz.allocation 
# df['min_variance_weights'] = port_min_variance_l1.weights 
# df['min_variance_allocation'] = port_min_variance_l1.allocation 
# df['equi_weight_weights'] = port_equi_weight.weights 
# df['equi_weight_allocation'] = port_equi_weight.allocation 
# df['long_only_weights'] = port_long_only .weights 
# df['long_only_allocation'] = port_long_only .allocation 
    
