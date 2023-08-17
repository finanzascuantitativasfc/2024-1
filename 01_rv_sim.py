# -*- coding: utf-8 -*-
"""
Created on Wed Aug 16 09:33:50 2023

@author: Meva
"""

import numpy as np
import pandas as pd
import matplotlib as mpl
import scipy
import matplotlib.pyplot as plt

degrees_freedom = 5
size = 10**6
random_variable_type = 'student' # normal student uniform

if random_variable_type == 'normal':
    x = np.random.standard_normal(size=10**6)
elif random_variable_type == 'student':
    x = np.random.standard_t(df=degrees_freedom, size=size)
# x = np.random.uniform(size=size)

mpl.pyplot.figure()
plt.hist(x,bins=100)
plt.title(random_variable_type)
plt.show()


