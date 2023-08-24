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


class simulator():
    
    # constructor
    def __init__(self, coeff, rv_type, size=10**6, decimals=5):
        self.coeff = coeff
        self.rv_type = rv_type
        self.size = size
        self.decimals = decimals
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
        self.str_title = self.rv_type
        if self.rv_type == 'normal':
            self.vector = np.random.standard_normal(self.size)
        elif self.rv_type == 'student':
            self.vector = np.random.standard_t(df=self.coeff, size=self.size)
            self.str_title = self.str_title + ' df=' + str(self.coeff)
        elif self.rv_type == 'uniform':
            self.vector = np.random.uniform(size=self.size)
        elif self.rv_type == 'exponential':
            self.vector = np.random.exponential(scale=self.coeff, size=self.size)
            self.str_title += ' scale=' + str(self.coeff)
        elif self.rv_type == 'chi-squared':
            self.vector = np.random.chisquare(df=self.coeff, size=self.size)
            self.str_title += ' df=' + str(self.coeff)
            
    def compute_stats(self):
        self.mean = st.tmean(self.vector)
        self.volatility = st.tstd(self.vector)
        self.skewness = st.skew(self.vector)
        self.kurtosis = st.kurtosis(self.vector)
        self.jb_stat = self.size/6 * (self.skewness**2 + 1/4*self.kurtosis**2)
        self.p_value = 1 - st.chi2.cdf(self.jb_stat, df=2)
        self.is_normal = (self.p_value > 0.05) # equivalently jb < 6
        
    def plot(self):
        self.str_title += '\n' + 'mean=' + str(np.round(self.mean,self.decimals)) \
            + ' | ' + 'volatility=' + str(np.round(self.volatility,self.decimals)) \
            + '\n' + 'skewness=' + str(np.round(self.skewness,self.decimals)) \
            + ' | ' + 'kurtosis=' + str(np.round(self.kurtosis,self.decimals)) \
            + '\n' + 'JB stat=' + str(np.round(self.jb_stat,self.decimals)) \
            + ' | ' + 'p-value=' + str(np.round(self.p_value,self.decimals)) \
            + '\n' + 'is_normal=' + str(self.is_normal)
        plt.figure()
        plt.hist(self.vector,bins=100)
        plt.title(self.str_title)
        plt.show()