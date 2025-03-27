# -*- coding: utf-8 -*-
"""
Created on Tue Jan 16 12:07:23 2024

@author: lcota
"""
#%% imports
import numpy as np
import cvxpy as cp
import scipy.sparse as sp

# Generate data for factor model.
# n = 600
n = 10 
m = 30
np.random.seed(1)
mu = np.abs(np.random.randn(n, 1))
Sigma_tilde = np.random.randn(m, m)
Sigma_tilde = Sigma_tilde.T.dot(Sigma_tilde)
D = sp.diags(np.random.uniform(0, 0.9, size=n))
F = np.random.randn(n, m)


# Factor model portfolio optimization.
w = cp.Variable(n)
f = cp.Variable(m)
gamma = cp.Parameter(nonneg=True)
Lmax = cp.Parameter()
ret = mu.T @ w
risk = cp.quad_form(f, Sigma_tilde) + cp.sum_squares(np.sqrt(D) @ w)
prob_factor = cp.Problem(
    cp.Maximize(ret - gamma * risk),
    [cp.sum(w) == 1, f == F.T @ w, cp.norm(w, 1) <= Lmax],
)

# Solve the factor model problem.
Lmax.value = 2
gamma.value = 0.1
prob_factor.solve(verbose=True)

print(prob_factor)