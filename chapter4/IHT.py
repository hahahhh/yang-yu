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
def hard_threshold(y, m):
    #    p = x.shape[0]
    t = np.sort(np.abs(y))[::-1]
    threshold = t[m - 1]
    j = (np.abs(y) < threshold)
    y[j] = 0
    return y


# Returns the value of the objective function
def f(x, phi, y):
    return 0.5 * math.pow(la.norm(x - phi @ y, 2), 2)


def iht(x, phi, k, iter_max, epsilon, verbose, x_star):
    # Length of original signal
    p = phi.shape[1]
    # Initial estimate
    y_new = np.zeros(p)
    phi_transpose = np.transpose(phi)
    diff_list, error_list = [1.0], [f(x, phi, y_new)]

    for i in range(iter_max):
        y_old = y_new
        # Compute gradient
        grad = phi_transpose @ (x - phi @ y_old)
        # Perform gradient step
        x_temp = y_old + grad
        # Perform hard thresholding step
        y_new = hard_threshold(x_temp, k)

        if (la.norm(y_new - y_old, 2) / la.norm(y_new, 2)) < epsilon:
            break

        # Keep track of solutions and objective values
        diff_list.append(la.norm(y_new - x_star, 2))
        error_list.append(f(x, phi, y_new))
        if verbose:
            print("iter# = " + str(i) + ", ||x_new - x_old||_2 = " + str(la.norm(y_new - y_old, 2)))

    return y_new, diff_list, error_list
