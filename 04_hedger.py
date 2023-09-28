# -*- coding: utf-8 -*-
"""
Created on Thu Sep 21 09:15:55 2023

@author: Meva
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.stats as st
import importlib
import scipy.optimize as op

# import our own files and reload
import capm
importlib.reload(capm)

# inputs
position_security = 'V'
position_delta_usd = 10 # in mn USD
benchmark = '^SPX'
# hedge_universe = ['AAPL','MSFT','NVDA','AMZN','GOOG','META','NFLX','SPY','XLK','XLF']
hedge_universe = ['BRK-B','JPM','V','MA','BAC','MS','GS','BLK','SPY','XLF']
regularisation = 0.01

# compute correlations
df = capm.dataframe_correlation_beta(benchmark, position_security, hedge_universe)

# computations
hedge_securities = ['MA','SPY']
hedger = capm.hedger(position_security, position_delta_usd, benchmark, hedge_securities)
hedger.compute_betas()
hedger.compute_hedge_weights(regularisation)

# variables
position_beta_usd = hedger.position_beta_usd
hedge_weights = hedger.hedge_weights
hedge_delta_usd = hedger.hedge_delta_usd
hedge_beta_usd = hedger.hedge_beta_usd
