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
position_security = 'NVDA'
position_delta_usd = 10 # in mn USD
benchmark = '^SPX'
hedge_securities = ['GOOG','AAPL','MSFT','SPY']

hedger = capm.hedger(position_security, position_delta_usd, benchmark, hedge_securities)
hedger.compute_betas()
hedger.compute_hedge_weights()
hedge_weights_exact = hedger.hedge_weights


# parameters
betas = hedger.hedge_betas
target_delta = hedger.position_delta_usd
target_beta = hedger.position_beta_usd

# define the function to minimise
def cost_function(x, betas, target_delta, target_beta):
    dimensions = len(x)
    deltas = np.ones([dimensions])
    f_delta = (np.transpose(deltas).dot(x).item() + target_delta)**2
    f_beta = (np.transpose(betas).dot(x).item() + target_beta)**2
    f = f_delta + f_beta
    return f

# initial condition
x0 = - target_delta / len(betas) * np.ones([len(betas),1])

# compute optimisation
optimal_result = op.minimize(fun=cost_function, x0=x0,\
                             args=(betas,target_delta,target_beta))
hedge_weights_optimize = optimal_result.x
    
# print
print('------')
print('Optimisation result:')
print(optimal_result)
print('------')
