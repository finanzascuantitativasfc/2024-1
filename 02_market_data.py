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

# inputs
ric = 'EWW'

directory = 'C:\\Users\\Meva\\.spyder-py3\\2024-1\\data\\' # hardcoded
path = directory + ric + '.csv' 
raw_data = pd.read_csv(path)
t = pd.DataFrame()
t['date'] = pd.to_datetime(raw_data['Date'], dayfirst=True)
t['close'] = raw_data['Close']
t.sort_values(by='date', ascending=True)
t['close_previous'] = t['close'].shift(1)
t['return_close'] = t['close']/t['close_previous'] - 1
t = t.dropna()
t = t.reset_index(drop=True)


# inputs
inputs = random_variables.simulation_inputs()
inputs.rv_type = ric + ' | real data'
# options: standard_normal normal student uniform exponential chi-squared
inputs.decimals = 5

# computations
sim = random_variables.simulator(inputs)
sim.vector = t['return_close'].values
sim.inputs.size  = len(sim.vector)
sim.str_title = sim.inputs.rv_type
sim.compute_stats()
sim.plot()
    
plt.figure()
t.plot(kind='line', x='date', y='close', grid=True, color='blue',\
        title='Timeseries of close prices for '+ric)
plt.show()

rics = []
is_normals = []
for file_name in os.listdir(directory):
    print('file_name = ' + file_name)
    ric = file_name.split('.')[0]
    if ric == 'ReadMe':
        continue
    # get dataframe
    path = directory + ric + '.csv'
    raw_data = pd.read_csv(path)
    t = pd.DataFrame()
    t['date'] = pd.to_datetime(raw_data['Date'], dayfirst=True)
    t['close'] = raw_data['Close']
    t.sort_values(by='date', ascending=True)
    t['close_previous'] = t['close'].shift(1)
    t['return_close'] = t['close']/t['close_previous'] - 1
    t = t.dropna()
    t = t.reset_index(drop=True)
    # computations
    sim = random_variables.simulator(inputs)
    sim.vector = t['return_close'].values
    sim.inputs.size  = len(sim.vector)
    sim.str_title = sim.inputs.rv_type
    sim.compute_stats()
    # generate lists
    rics.append(ric)
    is_normals.append(sim.is_normal)
df = pd.DataFrame()
df['ric'] = rics
df['is_normal'] = is_normals