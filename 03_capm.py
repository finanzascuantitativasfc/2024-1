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

# inputs
benchmark = 'XLV' # x
security = 'PG' # y

# compute Capital Asset Pricing Model
capm = market_data.capm(benchmark, security)
capm.synchronise_timeseries()
capm.plot_timeseries()
capm.compute_linear_regression()
capm.plot_linear_regression()
