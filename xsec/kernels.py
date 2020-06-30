"""
Definitions of the Gaussian process kernel functions. Large parts
adapted from the scikit-learn v0.19.2 source code, under the New BSD
License:

Copyright (c) 2007--2018 The scikit-learn developers.
All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are
met:

  a. Redistributions of source code must retain the above copyright
     notice, this list of conditions and the following disclaimer.
  b. Redistributions in binary form must reproduce the above copyright
     notice, this list of conditions and the following disclaimer in the
     documentation and/or other materials provided with the
     distribution.
  c. Neither the name of the Scikit-learn Developers  nor the names of
     its contributors may be used to endorse or promote products derived
     from this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS
IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED
TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A
PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE REGENTS OR
CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF
LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING
NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
"""

import math
import numpy as np
from scipy.special import kv, gamma
from scipy.spatial.distance import pdist, cdist, squareform


def _check_length_scale(X, length_scale):
    length_scale = np.squeeze(length_scale).astype(float)
    if np.ndim(length_scale) > 1:
        raise ValueError("length_scale cannot be of dimension greater than 1")
    if np.ndim(length_scale) == 1 and X.shape[1] != length_scale.shape[0]:
        raise ValueError(
            "Anisotropic kernel must have the same number of "
            "dimensions as data (%d!=%d)" % (length_scale.shape[0], X.shape[1])
        )
    return length_scale


def WhiteKernel(X, Y=None, noise_level=1.0):
    """
    Mainly used as part of a sum-kernel where it explains the
    noise-component of the signal.

    Return:
        k(x_1, x_2) = noise_level if x_1 == x_2 else 0
    """

    X = np.atleast_2d(X)

    if Y is None:
        K = noise_level * np.eye(X.shape[0])
        return K
    # Else, if Y is not None:
    return np.zeros((X.shape[0], Y.shape[0]))


def LinearKernel(X, Y=None, c=1.0):
    """
    Implementation of a linear kernel.

    The linear kernel is non-stationary. The parameter c of the linear
    kernel specifies the origin. The kernel is given by:
        k(x_i, x_j) = ((x_i - c) \cdot (x_j - c))
    """

    X = np.atleast_2d(X)
    c = _check_length_scale(X, c)

    # Rescale X
    Xsub = X - c

    if Y is None:
        K = np.inner(Xsub, Xsub)
    else:
        K = np.inner(Xsub, Y - c)

    return K


def Matern(X, Y=None, length_scale=1.0, nu=1.5):
    """
    Standard Matern kernel implementation, parametrised as in
    scikit-learn.
    """
    X = np.atleast_2d(X)
    length_scale = _check_length_scale(X, length_scale)
    if Y is None:
        dists = pdist(X / length_scale, metric="euclidean")
    else:
        dists = cdist(X / length_scale, Y / length_scale, metric="euclidean")

    if nu == 1.5:
        K = dists * math.sqrt(3)
        K = (1.0 + K) * np.exp(-K)
    elif nu == 0.5:
        K = np.exp((-1.0) * dists)
    elif nu == 2.5:
        K = dists * math.sqrt(5)
        K = (1.0 + K + K ** 2 / 3.0) * np.exp(-K)
    else:  # general case; expensive to evaluate
        K = dists
        K[K == 0.0] += np.finfo(float).eps  # strict zeros result in nan
        tmp = math.sqrt(2 * nu) * K
        K.fill((2 ** (1.0 - nu)) / gamma(nu))
        K *= tmp ** nu
        K *= kv(nu, tmp)

    if Y is None:
        # convert from upper-triangular matrix to square matrix
        K = squareform(K)
        np.fill_diagonal(K, 1)

    return K


def get_kernel(kernel_string):
    """
    Construct the kernel function object from a string containing its
    components and the relevant fitted hyperparameter values. In
    particular, the returned function takes X and Y (optional) as
    arguments.

    Parameters
    ----------
    kernel_string : str
        String representation of the kernel function with optimised
        hyperparameters.

    Returns
    -------
    kernel_function(X, Y=None) : function
        Kernel function object.
    """
    kernel_function = eval("lambda X, Y=None: " + kernel_string)

    return kernel_function
