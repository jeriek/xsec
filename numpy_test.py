#!/usr/bin/env python
# -*- coding: UTF-8 -*-

from __future__ import print_function

import numpy as np
# from time import time
import time

# Let's take the randomness out of random numbers (for reproducibility)
np.random.seed(0)

size = 5000
A, B = np.random.random((size, size)), np.random.random((size, size))

# Matrix multiplication
N = 10
t_wall = time.time()
t_CPU = time.clock()
for i in range(N):
    np.dot(A, B)
delta_wall = time.time() - t_wall
delta_CPU = time.clock() - t_CPU
print('Dotted two %dx%d matrices in %0.2f s. (wall time)' % (size, size, delta_wall / N))
print('Dotted two %dx%d matrices in %0.2f s. (CPU time)' % (size, size, delta_CPU / N))
del A, B