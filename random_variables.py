# -*- coding: utf-8 -*-
"""
Created on Wed Aug 23 09:35:55 2023

@author: Meva
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.stats as st
import importlib


class simulation_inputs:
    
    def __init__(self):
        self.df = None
        self.scale = None
        self.mean = None
        self.std = None
        self.rv_type = None
        self.size = None
        self.decimals = None

class simulator:
    
    # constructor
    def __init__(self, inputs):
        self.inputs = inputs
        self.str_title = None
        self.vector = None
        self.mean = None
        self.volatility = None
        self.skewness = None
        self.kurtosis = None
        self.jb_stat = None
        self.p_value = None
        self.is_normal = None
        
    def generate_vector(self):
        self.str_title = self.inputs.rv_type
        if self.inputs.rv_type == 'standard_normal':
            self.vector = np.random.standard_normal(self.inputs.size)
        elif self.inputs.rv_type == 'normal':
            self.vector = np.random.normal(self.inputs.mean, self.inputs.std, self.inputs.size)
        elif self.inputs.rv_type == 'student':
            self.vector = np.random.standard_t(df=self.inputs.df, size=self.inputs.size)
            self.str_title = self.str_title + ' df=' + str(self.inputs.df)
        elif self.inputs.rv_type == 'uniform':
            self.vector = np.random.uniform(size=self.inputs.size)
        elif self.inputs.rv_type == 'exponential':
            self.vector = np.random.exponential(scale=self.inputs.scale, size=self.inputs.size)
            self.str_title += ' scale=' + str(self.inputs.scale)
        elif self.inputs.rv_type == 'chi-squared':
            self.vector = np.random.chisquare(df=self.inputs.df, size=self.inputs.size)
            self.str_title += ' df=' + str(self.inputs.df)
            
    def compute_stats(self):
        self.mean = st.tmean(self.vector)
        self.volatility = st.tstd(self.vector)
        self.skewness = st.skew(self.vector)
        self.kurtosis = st.kurtosis(self.vector)
        self.jb_stat = self.inputs.size/6 * (self.skewness**2 + 1/4*self.kurtosis**2)
        self.p_value = 1 - st.chi2.cdf(self.jb_stat, df=2)
        self.is_normal = (self.p_value > 0.05) # equivalently jb < 6
        
    def plot(self):
        self.str_title += '\n' + 'mean=' + str(np.round(self.mean,self.inputs.decimals)) \
            + ' | ' + 'volatility=' + str(np.round(self.volatility,self.inputs.decimals)) \
            + '\n' + 'skewness=' + str(np.round(self.skewness,self.inputs.decimals)) \
            + ' | ' + 'kurtosis=' + str(np.round(self.kurtosis,self.inputs.decimals)) \
            + '\n' + 'JB stat=' + str(np.round(self.jb_stat,self.inputs.decimals)) \
            + ' | ' + 'p-value=' + str(np.round(self.p_value,self.inputs.decimals)) \
            + '\n' + 'is_normal=' + str(self.is_normal)
        plt.figure()
        plt.hist(self.vector,bins=100)
        plt.title(self.str_title)
        plt.show()