# -*- coding: utf-8 -*-
"""
Created on Thu Sep  7 09:11:35 2023

@author: Meva
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.stats as st
import importlib


def load_timeseries(ric):
    directory = 'C:\\Users\\Meva\\.spyder-py3\\2024-1\\data\\' # hardcoded
    path = directory + ric + '.csv' 
    raw_data = pd.read_csv(path)
    t = pd.DataFrame()
    t['date'] = pd.to_datetime(raw_data['Date'], dayfirst=True)
    t['close'] = raw_data['Close']
    t = t.sort_values(by='date', ascending=True)
    t['close_previous'] = t['close'].shift(1)
    t['return'] = t['close']/t['close_previous'] - 1
    t = t.dropna()
    t = t.reset_index(drop=True)
    return t

    
class distribution:
    
    # constructor
    def __init__(self, ric, decimals = 5):  
        self.ric = ric
        self.decimals = decimals
        self.str_title = None
        self.timeseries = None
        self.vector = None
        self.mean_annual = None
        self.volatility_annual = None
        self.sharpe_ratio = None
        self.var_95 = None
        self.skewness = None
        self.kurtosis = None
        self.jb_stat = None
        self.p_value = None
        self.is_normal = None
        
    def load_timeseries(self):
        self.timeseries = load_timeseries(self.ric)
        self.vector = self.timeseries['return'].values
        self.size = len(self.vector)
        self.str_title = self.ric + " | real data"
        
    def plot_timeseries(self):
        plt.figure()
        self.timeseries.plot(kind='line', x='date', y='close', grid=True, color='blue',\
                title='Timeseries of close prices for ' + self.ric)
        plt.show()
            
    def compute_stats(self, factor = 252):
        self.mean_annual = st.tmean(self.vector) * factor
        self.volatility_annual = st.tstd(self.vector) * np.sqrt(factor)
        self.sharpe_ratio = self.mean_annual / self.volatility_annual if self.volatility_annual > 0 else 0.0
        self.var_95 = np.percentile(self.vector,5)
        self.skewness = st.skew(self.vector)
        self.kurtosis = st.kurtosis(self.vector)
        self.jb_stat = self.size/6 * (self.skewness**2 + 1/4*self.kurtosis**2)
        self.p_value = 1 - st.chi2.cdf(self.jb_stat, df=2)
        self.is_normal = (self.p_value > 0.05) # equivalently jb < 6
        
    def plot_histogram(self):
        self.str_title += '\n' + 'mean_annual=' + str(np.round(self.mean_annual,self.decimals)) \
            + ' | ' + 'volatility_annual=' + str(np.round(self.volatility_annual,self.decimals)) \
            + '\n' + 'sharpe_ratio=' + str(np.round(self.sharpe_ratio,self.decimals)) \
            + ' | ' + 'var_95=' + str(np.round(self.var_95,self.decimals)) \
            + '\n' + 'skewness=' + str(np.round(self.skewness,self.decimals)) \
            + ' | ' + 'kurtosis=' + str(np.round(self.kurtosis,self.decimals)) \
            + '\n' + 'JB stat=' + str(np.round(self.jb_stat,self.decimals)) \
            + ' | ' + 'p-value=' + str(np.round(self.p_value,self.decimals)) \
            + '\n' + 'is_normal=' + str(self.is_normal)
        plt.figure()
        plt.hist(self.vector,bins=100)
        plt.title(self.str_title)
        plt.show()
        
