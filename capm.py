# -*- coding: utf-8 -*-
"""
Created on Thu Sep 21 09:12:43 2023

@author: Meva
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.stats as st
import scipy.optimize as op
import importlib
    
import market_data
importlib.reload(market_data)

def compute_beta(benchmark, security):
    m = model(benchmark, security)
    m.synchronise_timeseries()
    m.compute_linear_regression()
    return m.beta

def compute_correlation(security_1, security_2):
    m = model(security_1, security_2)
    m.synchronise_timeseries()
    m.compute_linear_regression()
    return m.correlation

def dataframe_correlation_beta(benchmark, position_security, hedge_universe):
    decimals = 5
    df = pd.DataFrame()
    correlations = []
    betas = []
    for hedge_security in hedge_universe:
        correlation = compute_correlation(position_security, hedge_security)
        beta = compute_beta(benchmark, hedge_security)
        correlations.append(np.round(correlation, decimals))
        betas.append(np.round(beta, decimals))
    df['hedge_security'] = hedge_universe
    df['correlation'] = correlations
    df['beta'] = betas
    df = df.dropna()
    df = df.sort_values(by='correlation', ascending=False)
    return df
    
# define the function to minimise for beta- and delta-neutral hedge
def cost_function_capm(x, betas, target_delta, target_beta, regularisation):
    dimensions = len(x)
    deltas = np.ones([dimensions])
    f_delta = (np.transpose(deltas).dot(x).item() + target_delta)**2
    f_beta = (np.transpose(betas).dot(x).item() + target_beta)**2
    f_penalty = regularisation*(np.sum(x**2))
    f = f_delta + f_beta + f_penalty
    return f

  
class model:
    
    # constructor
    def __init__(self, benchmark, security, decimals = 6):
        self.benchmark = benchmark
        self.security = security
        self.decimals = decimals
        self.timeseries = None
        self.x = None
        self.y = None
        self.alpha = None
        self.beta = None
        self.p_value = None
        self.null_hypothesis = None
        self.correlation = None
        self.r_squared = None
        self.predictor_linreg = None
        
    def synchronise_timeseries(self):
        self.timeseries = market_data.synchronise_timeseries(self.benchmark, self.security)
        
    def plot_timeseries(self):
        plt.figure(figsize=(12,5))
        plt.title('Time series of close prices')
        plt.xlabel('Time')
        plt.ylabel('Prices')
        ax = plt.gca()
        ax1 = self.timeseries.plot(kind='line', x='date', y='close_x', ax=ax, grid=True,\
                                  color='blue', label=self.benchmark)
        ax2 = self.timeseries.plot(kind='line', x='date', y='close_y', ax=ax, grid=True,\
                                  color='red', secondary_y=True, label=self.security)
        ax1.legend(loc=2)
        ax2.legend(loc=1)
        plt.show()
        
    def compute_linear_regression(self):
        # compute linear regression
        self.x = self.timeseries['return_x'].values
        self.y = self.timeseries['return_y'].values
        slope, intercept, r_value, p_value, std_err = st.linregress(self.x,self.y)
        self.alpha = np.round(intercept, self.decimals)
        self.beta = np.round(slope, self.decimals)
        self.p_value = np.round(p_value, self.decimals) 
        self.null_hypothesis = p_value > 0.05 # p_value < 0.05 --> reject null hypothesis
        self.correlation = np.round(r_value, self.decimals) # correlation coefficient
        self.r_squared = np.round(r_value**2, self.decimals) # pct of variance of y explained by x
        self.predictor_linreg = intercept + slope*self.x
        
    def plot_linear_regression(self):
        self.x = self.timeseries['return_x'].values
        self.y = self.timeseries['return_y'].values
        str_self = 'Linear regression | security ' + self.security\
            + ' | benchmark ' + self.benchmark + '\n'\
            + 'alpha (intercept) ' + str(self.alpha)\
            + ' | beta (slope) ' + str(self.beta) + '\n'\
            + 'p-value ' + str(self.p_value)\
            + ' | null hypothesis ' + str(self.null_hypothesis) + '\n'\
            + 'correl (r-value) ' + str(self.correlation)\
            + ' | r-squared ' + str(self.r_squared)
        str_title = 'Scatterplot of returns' + '\n' + str_self
        plt.figure()
        plt.title(str_title)
        plt.scatter(self.x,self.y)
        plt.plot(self.x, self.predictor_linreg, color='green')
        plt.ylabel(self.security)
        plt.xlabel(self.benchmark)
        plt.grid()
        plt.show()
        
        
class hedger:
    
    def __init__(self, position_security, position_delta_usd, benchmark, hedge_securities):
        self.position_security = position_security
        self.position_delta_usd = position_delta_usd
        self.position_beta = None
        self.position_beta_usd = None
        self.benchmark = benchmark
        self.hedge_securities = hedge_securities
        self.hedge_betas = []
        self.hedge_weights = None
        self.hedge_delta_usd = None
        self.hedge_cost_usd = None
        self.hedge_beta_usd = None
               
    def compute_betas(self):
        self.position_beta = compute_beta(self.benchmark, self.position_security)
        self.position_beta_usd = self.position_beta * self.position_delta_usd
        for security in self.hedge_securities:
            beta = compute_beta(self.benchmark, security)
            self.hedge_betas.append(beta)
            
    def compute_hedge_weights(self, regularisation = 0):
        # initial condition
        # x0 = - self.position_delta_usd / len(self.hedge_betas) * np.ones([len(self.hedge_betas),1])
        x0 = [- self.position_delta_usd / len(self.hedge_betas)] * len(self.hedge_betas)
        # compute optimisation
        optimal_result = op.minimize(fun=cost_function_capm, x0=x0,\
                                     args=(self.hedge_betas, \
                                           self.position_delta_usd, \
                                           self.position_beta_usd, \
                                           regularisation))
        self.hedge_weights = optimal_result.x
        self.hedge_delta_usd = np.sum(self.hedge_weights)
        self.hedge_cost_usd = np.sum(np.abs(self.hedge_weights))
        self.hedge_beta_usd = np.transpose(self.hedge_betas).dot(self.hedge_weights).item()        
            
    def compute_hedge_weights_exact(self):
        # exact solution using matrix algebra
        dimensions = len(self.hedge_securities)
        if dimensions != 2:
            print('------')
            print('Cannot compute exact solution because dimensions: ' + str(dimensions) + ' =/= 2')
            return
        deltas = np.ones([dimensions])
        mtx = np.transpose(np.column_stack((deltas,self.hedge_betas)))
        targets = -np.array([[self.position_delta_usd],[self.position_beta_usd]])
        self.hedge_weights = np.linalg.inv(mtx).dot(targets)
        self.hedge_delta_usd = np.sum(self.hedge_weights)
        self.hedge_beta_usd = np.transpose(self.hedge_betas).dot(self.hedge_weights).item()
        
        
        
        