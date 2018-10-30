#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2018/10/29 21:43
# @Author  : Mamba
# @Site    : 
# @File    : IHT.py

import math
import numpy as np
from numpy import linalg as la


# Hard thresholding function
def hard_threshold(x, k):
    #    p = x.shape[0]
    t = np.sort(np.abs(x))[::-1]
    threshold = t[k - 1]
    j = (np.abs(x) < threshold)
    x[j] = 0
    return x


# Returns the value of the objective function
def f(y, phi, x):
    return 0.5 * math.pow(la.norm(y - phi @ x, 2), 2)


def iht(y, phi, k, iter_max, epsilon, verbose, x_star):
    # Length of original signal
    p = phi.shape[1]
    # Initial estimate
    x_new = np.zeros(p)
    phi_transpose = np.transpose(phi)
    x_list, f_list = [1.0], [f(y, phi, x_new)]

    for i in range(iter_max):
        x_old = x_new
        # Compute gradient
        grad = phi_transpose @ (y - phi @ x_old)
        # Perform gradient step
        x_temp = x_old + grad
        # Perform hard thresholding step
        x_new = hard_threshold(x_temp, k)

        if (la.norm(x_new - x_old, 2) / la.norm(x_new, 2)) < epsilon:
            break

        # Keep track of solutions and objective values
        x_list.append(la.norm(x_new - x_star, 2))
        f_list.append(f(y, phi, x_new))
        if verbose:
            print("iter# = " + str(i) + ", ||x_new - x_old||_2 = " + str(la.norm(x_new - x_old, 2)))

    # print("Number of steps:", len(f_list))
    return x_new, x_list, f_list
