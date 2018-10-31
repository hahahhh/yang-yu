#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2018/10/29 21:46
# @Author  : Mamba
# @File    : test_IHT.py


from unittest import TestCase
from numpy import linalg as la
import numpy as np
import random
from chapter4.IHT import iht


class TestIHT(TestCase):
    def test_IHT_noiseless(self):
        # Ambient dimension:
        p = 1000
        # Number of samples:
        n = 300
        # Sparsity level:
        m = 10
        # Generate a p-dimensional zero vector
        y_m = np.zeros(p)
        # Randomly sample k indices in the range [1:p]
        y_m_ind = random.sample(range(p), m)
        # Set x_star_ind with k random elements from Gaussian distribution
        y_m[y_m_ind] = np.random.randn(m)
        # Normalize
        y_m = (1 / la.norm(y_m, 2)) * y_m
        phi = (1 / np.sqrt(n * 1.0)) * np.random.randn(n, p)
        # Observation model
        x = phi @ y_m
        # Precision parameter
        epsilon = 1e-16
        iter_max = 100
        # Run algorithm:
        x_iht, x_list, f_list = iht(x, phi, m, iter_max, epsilon, False, y_m)
        est = x_iht[np.where(x_iht != 0)]
        real = y_m[np.where(x_iht != 0)]
        self.assertEqual(len(est), m)
        for i, real_value in enumerate(real):
            self.assertAlmostEqual(real_value, est[i])
        pass
