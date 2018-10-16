# Kernel module
# Adapted from scikit-learn v0.19.2 source code

import math
import numpy as np
from scipy.special import kv, gamma
from scipy.spatial.distance import pdist, cdist, squareform

def _check_length_scale(X, length_scale):
    length_scale = np.squeeze(length_scale).astype(float)
    if np.ndim(length_scale) > 1:
        raise ValueError("length_scale cannot be of dimension greater than 1")
    if np.ndim(length_scale) == 1 and X.shape[1] != length_scale.shape[0]:
        raise ValueError("Anisotropic kernel must have the same number of "
                         "dimensions as data (%d!=%d)"
                         % (length_scale.shape[0], X.shape[1]))
    return length_scale


def WhiteKernel(X, Y=None, noise_level=1.0):
    """
    Mainly used as part of a sum-kernel where it explains the noise-component of the signal. 
    Return: 
        k(x_1, x_2) = noise_level if x_1 == x_2 else 0
    """

    X = np.atleast_2d(X)

    if Y is None:
        K = noise_level * np.eye(X.shape[0])
        return K
    else:
        return np.zeros((X.shape[0], Y.shape[0]))



def MaternKernel(X, Y=None, length_scale=1.0, nu=1.5):
    """
    Standard Matern kernel implementation, parametrised as in scikit-learn.
    """
    X = np.atleast_2d(X)
    length_scale = _check_length_scale(X, length_scale)
    if Y is None:
        dists = pdist(X / length_scale, metric='euclidean')
    else:
        dists = cdist(X / length_scale, Y / length_scale,
                      metric='euclidean')

    if nu == 0.5:
        K = np.exp(-dists)
    elif nu == 1.5:
        K = dists * math.sqrt(3)
        K = (1. + K) * np.exp(-K)
    elif nu == 2.5:
        K = dists * math.sqrt(5)
        K = (1. + K + K ** 2 / 3.0) * np.exp(-K)
    else:  # general case; expensive to evaluate
        K = dists
        K[K == 0.0] += np.finfo(float).eps  # strict zeros result in nan
        tmp = (math.sqrt(2 * nu) * K)
        K.fill((2 ** (1. - nu)) / gamma(nu))
        K *= tmp ** nu
        K *= kv(nu, tmp)

    if Y is None:
        # convert from upper-triangular matrix to square matrix
        K = squareform(K)
        np.fill_diagonal(K, 1)

    return K