# -*- coding: utf-8 -*-
"""
Created on Wed Aug 16 09:33:50 2023

@author: Meva
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.stats as st

# inputs
coeff = 5
# df in student and chi-squared, scale in exponential
size = 10**6
random_variable_type = 'student'
# options: normal student uniform exponential chi-squared
decimals = 5


# code
str_title = random_variable_type
if random_variable_type == 'normal':
    x = np.random.standard_normal(size=10**6)
elif random_variable_type == 'student':
    x = np.random.standard_t(df=coeff, size=size)
    str_title = str_title + ' df=' + str(coeff)
elif random_variable_type == 'uniform':
    x = np.random.uniform(size=size)
elif random_variable_type == 'exponential':
    x = np.random.exponential(scale=coeff, size=size)
    str_title += ' scale=' + str(coeff)
elif random_variable_type == 'chi-squared':
    x = np.random.chisquare(df=coeff, size=size)
    str_title += ' df=' + str(coeff)

mu = st.tmean(x) # tmean
sigma = st.tstd(x) # tstd
skewness = st.skew(x)
kurtosis = st.kurtosis(x)

# Normality Test: Jarque-Bera
jb_stat = size/6 * (skewness**2 + 1/4*kurtosis**2)
p_value = 1 - st.chi2.cdf(jb_stat, df=2)
is_normal = (p_value > 0.05) # equivalently jb < 6

str_title += '\n' + 'mean=' + str(np.round(mu,decimals)) \
    + ' | ' + 'volatility=' + str(np.round(sigma,decimals)) \
    + '\n' + 'skewness=' + str(np.round(skewness,decimals)) \
    + ' | ' + 'kurtosis=' + str(np.round(kurtosis,decimals)) \
    + '\n' + 'JB stat=' + str(np.round(jb_stat,decimals)) \
    + ' | ' + 'p-value=' + str(np.round(p_value,decimals)) \
    + '\n' + 'is_normal=' + str(is_normal)

# plot
plt.figure()
plt.hist(x,bins=100)
plt.title(str_title)
plt.show()


####################################
# loop of Jarque-Bera normality test
####################################

n = 0
is_normal = True
str_title = 'normal'

while is_normal and n < 500:
    x = np.random.standard_normal(size=10**6)
    mu = st.tmean(x) # tmean
    sigma = st.tstd(x) # tstd
    skewness = st.skew(x)
    kurtosis = st.kurtosis(x)
    jb_stat = size/6 * (skewness**2 + 1/4*kurtosis**2)
    p_value = 1 - st.chi2.cdf(jb_stat, df=2)
    is_normal = (p_value > 0.05) # equivalently jb < 6
    print('n=' + str(n) + ' | is_normal=' + str(is_normal))
    n += 1

str_title += '\n' + 'mean=' + str(np.round(mu,decimals)) \
    + ' | ' + 'volatility=' + str(np.round(sigma,decimals)) \
    + '\n' + 'skewness=' + str(np.round(skewness,decimals)) \
    + ' | ' + 'kurtosis=' + str(np.round(kurtosis,decimals)) \
    + '\n' + 'JB stat=' + str(np.round(jb_stat,decimals)) \
    + ' | ' + 'p-value=' + str(np.round(p_value,decimals)) \
    + '\n' + 'is_normal=' + str(is_normal)

# plot
plt.figure()
plt.hist(x,bins=100)
plt.title(str_title)
plt.show()


###############
# fake homework
###############

# explanation of the homework goes here
# e.g. what confidence threshold to use, say 3.75%
# return is_normal: boolean for the result of Jarque-Bera test
import scipy.stats as st
def test_jarque_bera(x):
    # your code goes here
    
    return is_normal