# -*- coding: utf-8 -*-
"""
Created on Tue Oct 24 09:00:46 2023

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

# create offline instances of pca_model
notional = 1 # in mn USD
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

# rics = ['^MXX','^SPX','BTC-USD','MXNUSD=X',\
#         'XLK','XLF','XLV','XLP','XLY','XLE','XLI']
# rics = ['^MXX','^SPX','^IXIC', '^STOXX', '^GDAXI', '^FCHI', '^VIX',\
#         'BTC-USD', 'ETH-USD', 'SOL-USD', 'USDC-USD', 'USDT-USD', 'DAI-USD']
# rics = ['SPY','XLK','XLF','XLV','XLE','XLC',\
#         'XLY','XLP','XLI','XLB','XLRE','XLU']
# rics = ['BTC-USD', 'ETH-USD', 'SOL-USD', 'USDC-USD', 'USDT-USD', 'DAI-USD']
# rics = ['^SPX','IVW','IVE','QUAL','MTUM','SIZE','USMV',\
#             'XLK','XLF','XLV','XLP','XLY','XLI','XLC','XLU']
# rics = ['AAPL','MSFT','NVDA','AMZN','GOOG','META','NFLX',\
#         'BRK-B','JPM','V','MA','BAC','MS','GS','BLK',\
#         'LLY','JNJ','PG','MRK','ABBV','PFE']
# 0-6 \ 7-14 \ 15-20

df = market_data.synchronise_returns(rics)
mtx = df.drop(columns=['date'])
mtx_var_covar = np.cov(mtx, rowvar=False) * 252
mtx_correl = np.corrcoef(mtx, rowvar=False)

# min-var with eigenvectors
eigenvalues, eigenvectors = np.linalg.eigh(mtx_var_covar)
min_var_vector = eigenvectors[:,0]

# unit test for variance function
variance_1 = np.matmul(np.transpose(min_var_vector), np.matmul(mtx_var_covar, min_var_vector))

######################################
# min-var with scipy optimize minimize
######################################

# function to minimize
def portfolio_variance(x, mtx_var_covar):
    variance = np.matmul(np.transpose(x), np.matmul(mtx_var_covar, x))
    return variance

# compute optimisation
x0 = [1 / np.sqrt(len(rics))] * len(rics)
l2_norm = [{"type": "eq", "fun": lambda x: sum(x**2) - 1}] # unitary in norm L2
l1_norm = [{"type": "eq", "fun": lambda x: sum(abs(x)) - 1}] # unitary in norm L1
optimal_result = op.minimize(fun=portfolio_variance, x0=x0,\
                             args=(mtx_var_covar),\
                             constraints=(l2_norm))
optimize_vector = optimal_result.x
variance_2 = optimal_result.fun

df_weights = pd.DataFrame()
df_weights['rics'] = rics
df_weights['min_var_vector'] = min_var_vector
df_weights['optimize_vector'] = optimize_vector

