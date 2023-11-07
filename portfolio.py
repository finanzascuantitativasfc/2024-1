# -*- coding: utf-8 -*-
"""
Created on Wed Oct 25 09:18:52 2023

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
import capm
importlib.reload(capm)


def portfolio_variance(x, mtx_var_covar):
    variance = np.matmul(np.transpose(x), np.matmul(mtx_var_covar, x))
    return variance


class manager:
    
    def __init__(self, rics, notional):
        self.rics = rics
        self.notional = notional
        self.mtx_var_covar = None
        self.mtx_correl = None
        self.returns = None
        self.volatilities = None
        self.dataframe_metrics = None
        self.dataframe_timeseries = None
        
    def compute_covariance(self):
        decimals = 6
        factor = 252
        df = market_data.synchronise_returns(self.rics)
        mtx = df.drop(columns=['date'])
        self.mtx_var_covar = np.cov(mtx, rowvar=False) * factor
        self.mtx_correl = np.corrcoef(mtx, rowvar=False)
        returns = []
        volatilities = []
        for ric in self.rics:
            r = np.round(np.mean(df[ric]) * factor, decimals)
            v =  np.round(np.std(df[ric]) * np.sqrt(factor), decimals)
            returns.append(r)
            volatilities.append(v)
        self.returns = np.array(returns)
        self.volatilities = np.array(volatilities)
        df_m = pd.DataFrame()
        df_m['rics'] = self.rics
        df_m['returns'] = self.returns 
        df_m['volatilities'] = self.volatilities
        self.dataframe_metrics = df_m
        self.dataframe_timeseries = df

        
    def compute_portfolio(self, portfolio_type=None, target_return=None):
    
        # initial conditions
        x0 = [1 / len(self.rics)] * len(self.rics)
        
        # constraints and boundary conditions for the optimiser
        l1_norm = [{"type": "eq", "fun": lambda x: sum(abs(x)) - 1}] # unitary in norm L1
        l2_norm = [{"type": "eq", "fun": lambda x: sum(x**2) - 1}] # unitary in norm L2
        markowitz = [{"type": "eq", "fun": lambda x: self.returns.dot(x) - target_return}] # target return for markowitz
        
        # boundary conditions
        non_negative = [(0, None) for i in range(len(self.rics))]
        
        # compute optimal portfolios
        if portfolio_type == 'min_variance_l1':
            optimal_result = op.minimize(fun=portfolio_variance, x0=x0,\
                                         args=(self.mtx_var_covar),\
                                         constraints=(l1_norm))
            weights = np.array(optimal_result.x)
            
        elif portfolio_type == 'min_variance_l2':
            optimal_result = op.minimize(fun=portfolio_variance, x0=x0,\
                                         args=(self.mtx_var_covar),\
                                         constraints=(l2_norm))
            weights = np.array(optimal_result.x)
                       
        elif portfolio_type == 'long_only':
            optimal_result = op.minimize(fun=portfolio_variance, x0=x0,\
                                          args=(self.mtx_var_covar),\
                                          constraints=(l1_norm),\
                                          bounds=non_negative)
            weights = np.array(optimal_result.x)
            
        elif portfolio_type == 'markowitz':
            epsilon = 10**-4
            if target_return == None:
                target_return = np.mean(self.returns)
            elif target_return < np.min(self.returns):
                target_return = np.min(self.returns) + epsilon
            elif target_return > np.max(self.returns):
                target_return = np.max(self.returns) - epsilon
            optimal_result = op.minimize(fun=portfolio_variance, x0=x0,\
                                          args=(self.mtx_var_covar),\
                                          constraints=(l1_norm + markowitz),\
                                          bounds=non_negative)
            weights = np.array(optimal_result.x)
            
        else:
            portfolio_type = 'equi-weight'
            weights = np.array(x0)
        
        # fill output
        decimals = 6
        optimal_portfolio = output(self.rics, self.notional)
        optimal_portfolio.type = portfolio_type
        optimal_portfolio.weights = weights / sum(abs(weights))
        optimal_portfolio.allocation = self.notional * optimal_portfolio.weights
        optimal_portfolio.target_return = target_return
        optimal_portfolio.return_annual = np.round(self.returns.dot(weights), decimals)
        optimal_portfolio.volatility_annual = np.round(np.sqrt(\
                                              portfolio_variance(weights, self.mtx_var_covar))\
                                              , decimals)
        optimal_portfolio.sharpe_ratio =  optimal_portfolio.return_annual / optimal_portfolio.volatility_annual \
            if optimal_portfolio.volatility_annual > 0.0 else 0.0
        
        # extend dataframe of metrics with optimal weights and allocations
        df_al = self.dataframe_metrics.copy()
        df_al['weights'] = optimal_portfolio.weights
        df_al['allocation'] = optimal_portfolio.allocation
        optimal_portfolio.dataframe_allocation = df_al
        
        # extend dataframe of timeseries with porfolio returns
        df_ts = self.dataframe_timeseries.copy()
        rics = list(df_al['rics'])
        port_rets = df_ts[rics[0]].values * 0.0
        for ric in rics:
            df = df_al.loc[df_al['rics'] == ric]
            w = df['weights'].item()
            port_rets += df_ts[ric].values * w
        df_ts['portfolio'] = port_rets
        optimal_portfolio.dataframe_timeseries = df_ts
        
        # compute the remaining metrics for the optimal portfolio
        optimal_portfolio.compute_stats()
            
        return optimal_portfolio
        
        
class output:
    
    def __init__(self, rics, notional):
        self.rics = rics
        self.notional = notional
        self.type = None
        self.weights = None
        self.allocation = None
        self.target_return = None
        self.return_annual = None
        self.volatility_annual = None
        self.sharpe_ratio = None
        self.dataframe_allocation = None
        self.dataframe_timeseries = None
        self.var_95 = None
        self.skewness = None
        self.kurtosis = None
        self.jb_stat = None
        self.p_value = None
        self.is_normal = None
        
    def compute_stats(self):
        x = self.dataframe_timeseries['portfolio'].values
        self.var_95 = np.percentile(x,5)
        self.skewness = st.skew(x)
        self.kurtosis = st.kurtosis(x)
        self.jb_stat = len(x)/6 * (self.skewness**2 + 1/4*self.kurtosis**2)
        self.p_value = 1 - st.chi2.cdf(self.jb_stat, df=2)
        self.is_normal = (self.p_value > 0.05) # equivalently jb < 6
        
    def plot_histogram(self):
        decimals = 4
        str_title = 'Portfolio = ' + self.type
        if self.target_return != None:
            str_title += ' | target return = ' + str(np.round(self.target_return,decimals))
        str_title += '\n' + 'return_annual=' + str(np.round(self.return_annual,decimals)) \
            + ' | ' + 'volatility_annual=' + str(np.round(self.volatility_annual,decimals)) \
            + '\n' + 'sharpe_ratio=' + str(np.round(self.sharpe_ratio,decimals)) \
            + ' | ' + 'var_95=' + str(np.round(self.var_95,decimals)) \
            + '\n' + 'skewness=' + str(np.round(self.skewness,decimals)) \
            + ' | ' + 'kurtosis=' + str(np.round(self.kurtosis,decimals)) \
            + '\n' + 'JB stat=' + str(np.round(self.jb_stat,decimals)) \
            + ' | ' + 'p-value=' + str(np.round(self.p_value,decimals)) \
            + '\n' + 'is_normal=' + str(self.is_normal)
        plt.figure()
        x = self.dataframe_timeseries['portfolio'].values
        plt.hist(x,bins=100)
        plt.title(str_title)
        plt.show()
        
        