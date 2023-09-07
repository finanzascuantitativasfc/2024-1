# -*- coding: utf-8 -*-
"""
Created on Thu Aug 31 09:06:32 2023

@author: Meva
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.stats as st
import importlib
import os

# import our own files and reload
import random_variables
importlib.reload(random_variables)
import market_data
importlib.reload(market_data)

# inputs
directory = 'C:\\Users\\Meva\\.spyder-py3\\2024-1\\data\\' # hardcoded
ric = 'SPY'

# computations
dist = market_data.distribution(ric)
dist.load_timeseries()
dist.plot_timeseries()
dist.compute_stats()
dist.plot_histogram()
    

# loop to check normality in real distributions
rics = []
is_normals = []
for file_name in os.listdir(directory):
    print('file_name = ' + file_name)
    ric = file_name.split('.')[0]
    if ric == 'ReadMe':
        continue
    # compute stats
    dist = market_data.distribution(ric)
    dist.load_timeseries()
    dist.compute_stats()
    # generate lists
    rics.append(ric)
    is_normals.append(dist.is_normal)
df = pd.DataFrame()
df['ric'] = rics
df['is_normal'] = is_normals
df = df.sort_values(by='is_normal', ascending=False)
