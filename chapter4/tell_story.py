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

p = 1000  # Ambient dimension
n = 300   # Number of samples
k = 10   # Sparsity level

np.random.seed(1)
# Generate a p-dimensional zero vector
x_star = np.zeros(p)
# Randomly sample k indices in the range [1:p]
x_star_ind = random.sample(range(p),  k) 
# Set x_star_ind with k random elements from Gaussian distribution
x_star[x_star_ind] = np.random.randn(k)
# Normalize
x_star = (1 / la.norm(x_star, 2)) * x_star

# Plot
xs = range(p)
markerline, stemlines, baseline = plt.stem(xs, x_star, '-.')

# Generate sensing matrix
Phi = (1 / math.sqrt(n)) * np.random.randn(n, p)

# Observation model
y = Phi @ x_star

# Hard thresholding function
def hardThreshold(x, k):
#    p = x.shape[0]
    t = np.sort(np.abs(x))[::-1]    
    threshold = t[k-1]
    j = (np.abs(x) < threshold)
    x[j] = 0
    return x

# Returns the value of the objecive function
def f(y, A, x):
    return 0.5 * math.pow(la.norm(y - Phi @ x, 2), 2)


def IHT(y, A, k, iters, epsilon, verbose, x_star):
    # Length of original signal
    p = A.shape[1]
    # Length of measurement vector
#    n = A.shape[0]
    # Initial estimate
    x_new = np.zeros(p)    
    # Transpose of A
#    At = np.transpose(A)

    PhiT = np.transpose(Phi)
    
    x_list, f_list = [1.0], [f(y, Phi, x_new)]

    for i in range(iters):
        x_old = x_new
    
        # Compute gradient
        grad = PhiT @ (y - Phi @ x_old)
    
        # Perform gradient step
        x_temp = x_old + grad    
    
        # Perform hard thresholding step
        x_new = hardThreshold(x_temp, k)
    
        if (la.norm(x_new - x_old, 2) / la.norm(x_new, 2)) < epsilon:
            break
                
        # Keep track of solutions and objective values
        x_list.append(la.norm(x_new - x_star, 2))
        f_list.append(f(y, Phi, x_new))
        
        if verbose:
            print("iter# = "+ str(i) + ", ||x_new - x_old||_2 = " + str(la.norm(x_new - x_old, 2)))
    
    print("Number of steps:", len(f_list))
    return x_new, x_list, f_list
        
# Run algorithm
epsilon = 1e-16                # Precision parameter
iters = 100

x_IHT, x_list, f_list = IHT(y, Phi, k + 1, iters, epsilon, False, x_star)

markerline, stemlines, baseline = plt.stem(xs, x_IHT, '-.x')


np.random.seed(1)
L = 100
N = 1000
phi_matrix = np.resize(np.random.normal(size=L*N), (L, N))
np.linalg.matrix_rank(phi_matrix)
