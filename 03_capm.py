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
import capm
importlib.reload(capm)

# inputs
benchmark = '^SPX' # x
security = 'GOOG' # y

# compute Capital Asset Pricing Model
model = capm.model(benchmark, security)
model.synchronise_timeseries()
model.plot_timeseries()
model.compute_linear_regression()
model.plot_linear_regression()
