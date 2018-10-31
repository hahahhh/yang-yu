# -*- coding: utf-8 -*-
"""
Created on Mon Oct 29 21:33:30 2018

@author: JinZhu
"""

import math
import random
from numpy import linalg as la
import numpy as np
import matplotlib.pyplot as plt
from chapter4.IHT import iht

p = 1000  # Ambient dimension
n = 300   # Number of samples
k = 10   # Sparsity level

np.random.seed(1)
# Generate a p-dimensional zero vector
y_m = np.zeros(p)
# Randomly sample k indices in the range [1:p]
x_star_ind = random.sample(range(p),  k) 
# Set x_star_ind with k random elements from Gaussian distribution
y_m[x_star_ind] = np.random.randn(k)
# Normalize
y_m = (1 / la.norm(y_m, 2)) * y_m

# Plot
xs = range(p)
plt.stem(xs, y_m, '-.')

# Generate sensing matrix
Phi = (1 / math.sqrt(n)) * np.random.randn(n, p)

# Observation model
x = Phi @ y_m

# Run algorithm
# Precision parameter
epsilon = 1e-16
iter_max = 100

y_iht, diff_list, error_list = iht(x, Phi, k + 1, iter_max, epsilon, False, y_m)

plt.stem(xs, y_iht, '-.x')

# RIP matrix
np.random.seed(1)
L = 100
N = 1000
phi_matrix = np.resize(np.random.normal(size=L*N), (L, N))
np.linalg.matrix_rank(phi_matrix)
