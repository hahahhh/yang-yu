# -*- coding: utf-8 -*-
"""
Created on Mon Oct  1 18:34:38 2018

@author: Administrator
"""

import numpy as np
import unittest
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

def test_cd_linear_regression(self):
    np.random.seed(4)
    num = 3000
    dim = 2
    x = np.random.normal(0, 1, size=num*dim)
    x = np.reshape(x, (num, dim))
    inception = np.array([1.0 for i in range(num)])
    x = np.hstack([np.reshape(inception, (num, 1)), x])
    beta = np.array([1.0, 2.0, 3.0])
    beta = np.reshape(beta, (beta.shape[0], 1))
    y = np.matmul(x, beta) + np.random.normal(0, 1, num).reshape((num, 1))
    # np.matmul 等同于 np.dot
    beta_est = cd_linear_regression(y, x, 20)
    beta = beta.flatten()
    beta_est = beta_est.flatten()
    print("-------Test CD algorithm for LR----------")
    for i, single_beta in  enumerate(beta):
        self.assertAlmostEqual(single_beta, beta_est[i], places=1)
        print("True value: %s, Estimated value: %s" % (single_beta, beta_est[i]))
       

def square_l2_norm(a):
    a = np.reshape(a, (a.shape[0],1))
    return np.dot(a.T, a)


def cd_lasso(y, x, iter_max=100, lamda_value=100.0):
    feature_num = x.shape[1]
    beta = np.random.uniform(0, 1, feature_num).reshape((feature_num, 1))
    k = 1
    norm_arr = np.apply_along_axis(square_l2_norm, 0 ,x)
    norm_arr = norm_arr.flatten()
    while k < iter_max:
        last_beta = beta.copy()
        for i in range(feature_num):
            not_update_index = list(range(feature_num))
            not_update_index.pop(i)
            x_tmp = x[:, not_update_index]
            beta_tmp = beta[not_update_index].copy()
            beta_ols = (np.matmul(y[:, 0], x[:, i]) - 
                        np.sum(np.matmul(x_tmp*x[:, [i]], beta_tmp))) / norm_arr[i]
            threshold = lamda_value / norm_arr[i]
            #notice the usage of np.max
            beta[i] = np.sign(beta_ols) * np.max(np.array([np.abs(beta_ols) - threshold, 0]))
            pass
        if np.sqrt(np.sum(np.square(beta - last_beta))) < pow(10, -6):
            break
        else:
            pass
        return beta
    

def test_cd_lasso1():
    np.random.seed(4)
    num = 5000
    dim = 5
    x = np.random.normal(0, 1, size=num*dim)
    x = np.reshape(x, (num, dim))
    inception = np.array([1.0 for i in range(num)])
    x = np.hstack([np.reshape(inception, (num, 1)), x])
    beta = np.array([1.0, 2.0, 3.0, 0.00, 0.00, 0.2])
    beta = np.reshape(beta, (beta.shape[0], 1))
    y = np.matmul(x, beta) + np.random.normal(0, 1, num).reshape((num, 1))
    # np.matmul 等同于 np.dot
    beta_est = cd_lasso(y, x, iter_max=20, lamda_value=20.0)
    beta = beta.flatten()
    beta_est = beta_est.flatten()
    print("-------Test CD algorithm for Lasso----------")
    for i, single_beta in  enumerate(beta):
        #self.assertAlmostEqual(single_beta, beta_est[i], places=1)
        print("True value: %s, Estimated value: %s" % (single_beta, beta_est[i]))
    
    
    
 def cd_elasticNet(y, x, iter_max=100, lamda_value=100.0, alpha_value=0.5
                   ):
    feature_num = x.shape[1]
    beta = np.random.uniform(0, 1, feature_num).reshape((feature_num, 1))
    k = 1
    norm_arr = np.apply_along_axis(square_l2_norm, 0 ,x)
    norm_arr = norm_arr.flatten()
    while k < iter_max:
        last_beta = beta.copy()
        for i in range(feature_num):
            not_update_index = list(range(feature_num))
            not_update_index.pop(i)
            x_tmp = x[:, not_update_index]
            beta_tmp = beta[not_update_index].copy()
            beta_par = (np.matmul(y[:, 0], x[:, i]) - 
                        np.sum(np.matmul(x_tmp*x[:, [i]], beta_tmp))) / (norm_arr[i] + lamda_value*(1-alpha_value))
            threshold = (lamda_value*alpha_value) / (norm_arr[i] + lamda_value*(1-alpha_value))
            #notice the usage of np.max
            beta[i] = np.sign(beta_par) * np.max(np.array([np.abs(beta_par) - threshold, 0]))
            pass
        if np.sqrt(np.sum(np.square(beta - last_beta))) < pow(10, -6):
            break
        else:
            pass
        return beta
    

def test_cd_elasticNet(alpha):
    np.random.seed(4)
    num = 5000
    dim = 5
    x1 = np.random.normal(0, 1, size=num).reshape((num, 1))
    x2 = 2*x1 + np.random.normal(0, 0.001, size=num).reshape((num, 1))
    x3_5 = np.random.normal(0, 1, size=num*3)
    x3_5 = np.reshape(x3_5, (num, 3))
    inception = np.array([1.0 for i in range(num)])
    x = np.hstack([np.reshape(inception, (num, 1)), x1, x2, x3_5])
    beta = np.array([1.0, 2.0, 3.0, 0.00, 0.00, 0.2])
    beta = np.reshape(beta, (beta.shape[0], 1))
    y = np.matmul(x, beta) + np.random.normal(0, 1, num).reshape((num, 1))
    # np.matmul 等同于 np.dot
    beta_est = cd_elasticNet(y, x, iter_max=20, lamda_value=20.0, alpha_value=alpha)
    beta = beta.flatten()
    beta_est = beta_est.flatten()
    print("-------Test CD algorithm for Lasso----------")
    for i, single_beta in  enumerate(beta):
        #self.assertAlmostEqual(single_beta, beta_est[i], places=1)
        print("True value: %s, Estimated value: %s" % (single_beta, beta_est[i]))   
    
    
    
    
    
    