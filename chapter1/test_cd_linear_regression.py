#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2018/9/13 18:57
# @Author  : Mamba
# @Site    : ${SITE}
# @File    : test_cd_linear_regression.py


from unittest import TestCase
import numpy as np
from chapter1.CDLinerRegression import cd_linear_regression, cd_lasso, cd_elastic_net
from sklearn.linear_model import ElasticNet
from sklearn.datasets import make_regression


class TestCDLinearRegression(TestCase):
    def test_cd_lasso1(self):
        np.random.seed(4)
        num = 5000
        dim = 5
        x = np.random.normal(0, 1, size=num*dim)
        x = np.reshape(x, (num, dim))
        intercept = np.array([1.0 for i in range(num)])
        x = np.hstack([np.reshape(intercept, (num, 1)), x])
        # beta = np.array([1.0, 2.0, 3.0, 0.02, -0.02, 0.2])
        beta = np.array([1.0, 2.0, 3.0, 0.00, 0.00, 0.2])
        beta = np.reshape(beta, (beta.shape[0], 1))
        y = np.matmul(x, beta) + np.random.normal(0, 1, num).reshape((num, 1))
        beta_est = cd_lasso(y, x, iter_max=20, lambda_value=20.0)

        beta = beta.flatten()
        beta_est = beta_est.flatten()
        print("--------- Test CD algorithm for Lasso ---------")
        for i, single_beta in enumerate(beta):
            self.assertAlmostEqual(single_beta, beta_est[i], places=1)
            print("True value: %s, Estimated value: %s" % (single_beta, beta_est[i]))

    def test_cd_linear_regression(self):
        np.random.seed(4)
        num = 3000
        dim = 2
        x = np.random.normal(0, 1, size=num*dim)
        x = np.reshape(x, (num, dim))
        intercept = np.array([1.0 for i in range(num)])
        x = np.hstack([np.reshape(intercept, (num, 1)), x])
        beta = np.array([1.0, 2.0, 3.0])
        beta = np.reshape(beta, (beta.shape[0], 1))
        y = np.matmul(x, beta) + np.random.normal(0, 1, num).reshape((num, 1))
        beta_est = cd_linear_regression(y, x, 20)
        beta = beta.flatten()
        beta_est = beta_est.flatten()
        print("--------- Test CD algorithm for LR ---------")
        for i, single_beta in enumerate(beta):
            self.assertAlmostEqual(single_beta, beta_est[i], places=1)
            print("True value: %s, Estimated value: %s" % (single_beta, beta_est[i]))

    # def test_cd_elastic_net(self):
    #     x, y, coef = make_regression(n_samples=1000, n_features=10,
    #                                  n_informative=6, random_state=4,
    #                                  coef=True)
    #     print("True coef: ", coef.tolist())
    #     sklearn_reg = ElasticNet(random_state=0, alpha=1.0, l1_ratio=0.5)
    #     sklearn_reg.fit(x, y)
    #     sklearn_fit_coef = sklearn_reg.coef_
    #     print("sklearn fitted coef: ", sklearn_fit_coef.tolist())
    #     # your need to solve the elastic net with coordinate descent algorithm:
    #     your_fit_coef = cd_elastic_net(y, x, iter_max=20, alpha=1.0, rho=0.5)
    #     print("sklearn fitted coef: ", your_fit_coef.tolist())
    #     for i, single_beta in enumerate(sklearn_fit_coef):
    #         self.assertAlmostEqual(single_beta, your_fit_coef[i], places=1)
    #         print("True value: %s, Estimated value: %s" % (single_beta, your_fit_coef[i]))
