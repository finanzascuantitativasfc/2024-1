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
inputs = random_variables.simulation_inputs()
inputs.df = 23 # degrees of freedom or df in student and chi-squared
inputs.scale = 17 # scale in exponential
inputs.mean = 5 # mean in normal
inputs.std = 10 # standard deviation or std in normal
inputs.size = 10**6
inputs.rv_type = 'student'
# options: standard_normal normal student uniform exponential chi-squared
inputs.decimals = 5

# computations
sim = random_variables.simulator(inputs)
sim.generate_vector()
sim.compute_stats()
sim.plot()