# -*- coding: utf-8 -*-
"""
Created on Thu Nov 16 09:15:26 2023

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
import options
importlib.reload(options)

inputs = options.inputs()
inputs.price = 36820 # S
inputs.time = 0 # t
inputs.maturity = 3/12 # T
inputs.strike = 25000 # K
inputs.interest_rate = 0.0453 # r
inputs.volatility = 0.556 # sigma
inputs.type = 'call'
inputs.monte_carlo_size = 10**6

option_mgr = options.manager(inputs)
option_mgr.compute_black_scholes_price()
option_mgr.compute_monte_carlo_price()
option_mgr.plot_histogram()

