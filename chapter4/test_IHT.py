#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2018/10/29 21:46
# @Author  : Mamba
# @Site    : ${SITE}
# @File    : test_IHT.py


from unittest import TestCase
import math
import random
from numpy import linalg as la
import numpy as np
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
        x_star = np.zeros(p)
        # Randomly sample k indices in the range [1:p]
        x_star_ind = random.sample(range(p), m)
        # Set x_star_ind with k random elements from Gaussian distribution
        x_star[x_star_ind] = np.random.randn(m)
        # Normalize
        x_star = (1 / la.norm(x_star, 2)) * x_star
        phi = (1 / math.sqrt(n)) * np.random.randn(n, p)
        # Observation model
        x = phi @ x_star
        # Run algorithm:
        epsilon = 1e-16  # Precision parameter
        iter_max = 100
        x_iht, x_list, f_list = iht(x, phi, m, iter_max, epsilon, False, x_star)
        est = x_iht[np.where(x_iht != 0)]
        real = x_star[np.where(x_iht != 0)]
        self.assertEqual(len(est), m)
        for i, real_value in enumerate(real):
            self.assertAlmostEqual(real_value, est[i])
        pass
