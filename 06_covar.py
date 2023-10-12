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
# rics = ['SPY','XLK','XLF','XLV','XLE','XLC',\
#         'XLY','XLP','XLI','XLB','XLRE','XLU']
rics = ['BTC-USD', 'ETH-USD', 'SOL-USD', 'USDC-USD', 'USDT-USD', 'DAI-USD']
# rics = ['^SPX','IVW','IVE','QUAL','MTUM','SIZE','USMV',\
#             'XLK','XLF','XLV','XLP','XLY','XLI','XLC','XLU']
# rics = ['AAPL','MSFT','NVDA','AMZN','GOOG','META','NFLX',\
#         'BRK-B','JPM','V','MA','BAC','MS','GS','BLK',\
#         'LLY','JNJ','PG','MRK','ABBV','PFE']
# 0-6 \ 7-14 \ 15-20


# synchronise all the timeseries of returns
df = market_data.synchronise_returns(rics)

# compute the variance-covariance and correlation matrices
mtx = df.drop(columns=['date'])
mtx_var_covar = np.cov(mtx, rowvar=False) * 252
mtx_correl = np.corrcoef(mtx, rowvar=False)

# unit test for correlations
# correl = capm.compute_correlation('^SPX','BTC-USD')

# compute eigenvectors and eigenvalues
eigenvalues, eigenvectors = np.linalg.eigh(mtx_var_covar)
variance_explained = eigenvalues / np.sum(eigenvalues)
prod = np.matmul(eigenvectors, np.transpose(eigenvectors))

##########################
# PCA for 2D visualisation
##########################

# compute min and max volatilities
volatility_min = np.sqrt(eigenvalues[0])
volatility_max = np.sqrt(eigenvalues[-1])

# compute PCA base for 2D visualisation
pca_vector_1 = eigenvectors[:,-1]
pca_vector_2 = eigenvectors[:,-2]
pca_eigenvalue_1 = eigenvalues[-1]
pca_eigenvalue_2 = eigenvalues[-2]
pca_variance_explained = variance_explained[-2:].sum()

# compute min variance portfolio
min_var_vector = eigenvectors[:,0]
min_var_eigenvalue = eigenvalues[0]
min_var_variance_explained = variance_explained[0]

