import math
import numpy as np
from numba import njit


@njit
def binomial_tree(x, N, u, d):
    us = u ** np.arange(N)
    ds = d ** np.arange(N)
    M = np.zeros((N, N))
    M[0, 0] = 1.0
    for j in range(N):
        for i in range(j + 1):
            M[i, j] = us[j - i] * ds[i]
    return M * x


def crr_binomial_tree(x, N, sigma, dt):
    u = np.exp(sigma * np.sqrt(dt))
    return binomial_tree(x, N, u, 1 / u)


def jr_binomial_tree(x, N, r, sigma, dt):
    m = np.exp(r * dt)
    s = np.exp(sigma * np.sqrt(dt))
    u = m * s
    d = m / s
    return binomial_tree(x, N, u, d)


def crr_ud(sigma, dt):
    return np.exp(sigma * np.sqrt(dt)), np.exp(-sigma * np.sqrt(dt))


def jr_ud(r, sigma, dt):
    m = np.exp((r - sigma**2 / 2) * dt)
    s = np.exp(sigma * np.sqrt(dt))
    u = m * s
    d = m / s
    return u, d


def crr_q(r, u, d, dt):
    R = np.exp(r * dt)
    return (R - d) / (u - d)


def jr_q():
    return 0.5


class BinomialTree:
    def __init__(self, r, sigma, T, dt):
        self.r = r
        self.sigma = sigma
        self.T = T
        self.dt = dt
        self.u, self.d = 2, 0.5

    def lattice(self, s0):
        return binomial_tree(s0, self.n_periods, self.u, self.d)

    @property
    def q(self):
        return 0.5

    @property
    def n_periods(self):
        return math.floor(self.T / self.dt)


class CRRModel(BinomialTree):
    def __init__(self, r, sigma, T, dt):
        super().__init__(r, sigma, T, dt)
        self.u, self.d = crr_ud(sigma, dt)

    @property
    def q(self):
        return crr_q(self.r, self.u, self.d, self.dt)


class JRModel(BinomialTree):
    def __init__(self, r, sigma, T, dt):
        super().__init__(r, sigma, T, dt)
        self.u, self.d = jr_ud(r, sigma, dt)

    @property
    def q(self):
        return jr_q()
