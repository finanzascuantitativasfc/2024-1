# -*- coding: utf-8 -*-
"""
Created on Wed Aug 16 09:33:50 2023

@author: Meva
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.stats as st
import importlib

# import our own files and reload
import random_variables
importlib.reload(random_variables)

# inputs
coeff = 5
# df in student and chi-squared, scale in exponential
size = 10**6
random_variable_type = 'normal'
# options: normal student uniform exponential chi-squared
decimals = 5

sim = random_variables.simulator(coeff, random_variable_type)
sim.generate_vector()
sim.compute_stats()
sim.plot()