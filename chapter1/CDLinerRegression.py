#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2018/9/13 17:44
# @Author  : Mamba
# @Site    : 
# @File    : CDLinerRegression.py


import numpy as np


def cd_linear_regression(y, x, iter_max=100):
    feature_num = x.shape[1]
    beta = np.random.uniform(0, 1, feature_num).reshape((feature_num, 1))
    k = 1
    while k < iter_max:
        last_beta = beta.copy()
        for i in range(feature_num):
            not_update_index = list(range(feature_num))
            not_update_index.pop(i)
            x_tmp = x[:, not_update_index]
            beta_tmp = beta[not_update_index].copy()
            beta[i] = (np.matmul(y[:, 0], x[:, i]) -
                       np.sum(np.matmul(x_tmp*x[:, [i]], beta_tmp))) / np.sum(np.square(x[:, i]))
            pass
        if np.sqrt(np.sum(np.square(beta - last_beta))) < pow(10, -6):
            break
        else:
            pass
    return beta


def square_l2_norm(z):
    return np.sum(np.square(z))


def cd_lasso(y, x, iter_max=100, lambda_value=100.0):
    feature_num = x.shape[1]
    beta = np.random.uniform(0, 1, feature_num).reshape((feature_num, 1))
    k = 1
    norm_arr = np.apply_along_axis(square_l2_norm, 0, x)
    while k < iter_max:
        last_beta = beta.copy()
        for i in range(feature_num):
            not_update_index = list(range(feature_num))
            not_update_index.pop(i)
            x_tmp = x[:, not_update_index]
            beta_tmp = beta[not_update_index].copy()
            beta_ols = (np.matmul(y[:, 0], x[:, i]) - np.sum(np.matmul(x_tmp*x[:, [i]], beta_tmp))) / norm_arr[i]
            threshold = lambda_value / norm_arr[i]
            # notice the usage of np.max
            beta[i] = np.sign(beta_ols) * np.max(np.array([np.abs(beta_ols) - threshold, 0]))
            pass
        if np.sqrt(np.sum(np.square(beta - last_beta))) < pow(10, -6):
            break
        else:
            pass
    return beta


def cd_ridge(y, x, iter_max=100, lambda_value=100.0):
    lambda_value = lambda_value
    feature_num = x.shape[1]
    beta = np.random.uniform(0, 1, feature_num).reshape((feature_num, 1))
    k = 1
    norm_arr = np.apply_along_axis(square_l2_norm, 0, x)
    while k < iter_max:
        last_beta = beta.copy()
        for i in range(feature_num):
            not_update_index = list(range(feature_num))
            not_update_index.pop(i)
            x_tmp = x[:, not_update_index]
            beta_tmp = beta[not_update_index].copy()
            beta_ols = (np.matmul(y[:, 0], x[:, i]) - np.sum(np.matmul(x_tmp*x[:, [i]], beta_tmp))) / norm_arr[i]
            beta[i] = beta_ols / (1.0 + lambda_value)
            pass
        if np.sqrt(np.sum(np.square(beta - last_beta))) < pow(10, -6):
            break
        else:
            pass
    return beta


def cd_elastic_net(y, x, iter_max=100, alpha=1.0, rho=0.5):
    feature_num = x.shape[1]
    beta = np.random.uniform(0, 1, feature_num)
    return beta
