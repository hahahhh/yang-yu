#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2018/10/8 15:40
# @Author  : Mamba
# @Site    : ${SITE}
from unittest import TestCase

from chapter2.svm import svm
import numpy as np


# @File    : test_svm.py
class TestSvm(TestCase):
    def test_svm(self):
        np.random.seed(1)
        num = 30
        cov_mat = np.diag([1, 1])
        x1 = np.random.multivariate_normal(mean=[5, 5], cov=cov_mat, size=num)
        x2 = np.random.multivariate_normal(mean=[-5, -5], cov=cov_mat, size=num)
        x = np.vstack([x1, x2])
        y1 = np.ones(num)
        y2 = -np.ones(num)
        y = np.append(y1, y2)
        beta, beta0 = svm(y, x, 1.0, 100)
        y_predict = np.sign(x.dot(beta) + beta0)
        self.assertTrue(np.all(y_predict == y))
