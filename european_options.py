import numpy as np
from numba import njit

from binomial_trees import BinomialTree


@njit
def rn_expectation(u, d, r, q, dt):
    return (q * u + (1 - q) * d) * np.exp(-r * dt)


@njit
def compute_induction(N, V, r, q, dt):
    for j in range(N - 1, 0, -1):
        for i in range(j):
            V[i, j - 1] = rn_expectation(V[i, j], V[i + 1, j], r, q, dt)
    return V


class EuropeanOption:
    def __init__(self, model: BinomialTree):
        self.model = model

    def payoff(self, x):
        return x

    def npv(self):
        lattice = self.model.lattice()
        N = self.model.n_periods
        terminal_value = lattice[:, -1]
        r = self.model.r
        dt = self.model.dt
        q = self.model.q
        V = np.zeros((N, N))
        V[:, -1] = self.payoff(terminal_value)
        return compute_induction(N, V, r, q, dt)


class EuropeanCallOption(EuropeanOption):
    def __init__(self, model, K_strike):
        self.model = model
        self.K_strike = K_strike

    def payoff(self, x):
        return np.maximum(x - self.K_strike, 0)


class EuropeanPutOption(EuropeanOption):
    def __init__(self, model, K_strike):
        self.model = model
        self.K_strike = K_strike

    def payoff(self, x):
        return np.maximum(self.K_strike - x, 0)
