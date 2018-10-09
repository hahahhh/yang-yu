#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2018/10/8 10:45
# @Author  : Mamba
# @Site    : 
# @File    : svm.py

import numpy as np


def svm(y, x, cost=1.0, iter_max=100):
    num = x.shape[0]
    p = x.shape[1]
    # compute kernel (actually, this step may be redundant):
    kernel = svm_kernel(x, num)
    # compute alpha:
    alpha = smo(y, x, kernel, cost, num, p, iter_max)
    # compute beta:
    beta = svm_beta(y, x, alpha=alpha, num=num, p=p)
    # compute intercept:
    beta0 = svm_intercept(y, x, beta, alpha)
    return beta, beta0


def svm_kernel(x, num):
    kernel = np.zeros([num, num])
    for i in range(num):
        for j in range(num):
            kernel[i, j] = np.matmul(x[i, :], x[j, :])
            pass
        pass
    return kernel


def svm_intercept(y, x, beta, alpha):
    sv_number = 0
    value = 0.0
    for index, alpha_value in enumerate(alpha):
        if alpha_value > 0:
            value += (1 - y[index] * np.matmul(x[index, :], beta)) / y[index]
            sv_number += 1
        pass
    return value / sv_number


def svm_beta(y, x, alpha, num, p):
    beta = np.zeros(p)
    for i in range(num):
        beta += alpha[i] * y[i] * x[i, :]
        pass
    return beta


def smo(y, x, kernel, C, num, p, iter_max, tol=10 ** -6):
    b = 0.0
    alpha = np.zeros(num)
    fxi_array = fxi_batch(num, y, alpha, kernel, b)
    error = fxi_array - y
    not_stop = True
    iter_num = 0
    while not_stop:
        iter_num += 1
        index1 = np.random.permutation(range(num))[0:2]
        index2 = index1[1]
        index1 = index1[0]
        alpha_o1 = alpha[index1]
        alpha_o2 = alpha[index2]
        # original value:
        target_value_old = target_value(y[index1], y[index2], alpha_o1, alpha_o2,
                                        kernel11=kernel[index1, index1], kernel22=kernel[index2, index2],
                                        kernel12=kernel[index1, index2], fit1=fxi_array[index1], fit2=fxi_array[index2])
        # update alpha_2:
        upper_bound_value = upper_bound(y[index1], y[index2], alpha_o1, alpha_o2, C)
        lower_bound_value = lower_bound(y[index1], y[index2], alpha_o1, alpha_o2, C)
        if upper_bound_value <= lower_bound_value:
            continue
        k = kernel[index1, index1] + kernel[index2, index2] - 2 * kernel[index1, index2]
        # avoid the same input point:
        if k < 0:
            continue
        alpha[index2] += index2 * (error[index1] - error[index2]) / k
        alpha[index2] = np.clip(alpha[index2], lower_bound_value, upper_bound_value)

        # update alpha_1:
        alpha[index1] += (alpha_o2 - alpha[index2]) * y[index1] * y[index2]
        # update fitting and error:
        fxi_array[index1] = fxi(index1, y, alpha, kernel, b)
        fxi_array[index2] = fxi(index2, y, alpha, kernel, b)
        error[index1] = fxi_array[index1] - y[index1]
        error[index2] = fxi_array[index2] - y[index2]
        # The threshold b is re-computed so that the KKT conditions are fulfilled for both optimized examples.
        b1 = b - error[index1] - y[index1] * kernel[index1, index1] * (alpha[index1] - alpha_o1) - \
             y[index2] * kernel[index1, index2] * (alpha[index2] - alpha_o2)
        b2 = b - error[index2] - y[index1] * kernel[index1, index2] * (alpha[index1] - alpha_o1) - \
             y[index2] * kernel[index2, index2] * (alpha[index2] - alpha_o2)
        if (alpha[index1] > 0) and (alpha[index1] < C):
            b = b1
        elif (alpha[index2] > 0) and (alpha[index2] < C):
            b = b2
        else:
            b = np.mean([b1, b2])
            pass

        # stop criterion:
        target_value_new = target_value(y[index1], y[index2], alpha[index1], alpha[index2],
                                        kernel11=kernel[index1, index1], kernel22=kernel[index2, index2],
                                        kernel12=kernel[index1, index2],
                                        fit1=fxi_array[index1], fit2=fxi_array[index2])
        if abs(target_value_new - target_value_old) < tol and (iter_num > iter_max):
            not_stop = False
        pass
    return alpha


def fxi(i, y, alpha, kernel, b):
    fxi_value = b
    fxi_value += np.sum(y * alpha * kernel[i, :])
    return fxi_value


def fxi_batch(num, y, alpha, kernel, b):
    fxi_list = np.zeros(num)
    for i in range(num):
        fxi_list[i] = fxi(i, y, alpha, kernel, b)
    return fxi_list


def lower_bound(y1, y2, alpha_o1, alpha_o2, cost):
    return max([0.0, alpha_o2 + alpha_o1 - cost]) if y1 * y2 == 1 else max([0.0, alpha_o2 - alpha_o1])


def upper_bound(y1, y2, alpha_o1, alpha_o2, cost):
    return min([cost, alpha_o2 + alpha_o1]) if y1 * y2 == 1 else min([cost, cost + alpha_o2 - alpha_o1])


def target_value(y1, y2, alpha1, alpha2, kernel11, kernel22, kernel12, fit1, fit2):
    return alpha1 + alpha2 - \
           0.5 * kernel11 * pow(alpha1, 2) - 0.5 * kernel22 * pow(alpha2, 2) - \
           y1 * y2 * kernel12 * alpha1 * alpha2 - \
           y1 * alpha1 * fit1 - y2 * alpha2 * fit2
