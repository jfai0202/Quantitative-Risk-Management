#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import sys
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.stats import t

df = pd.read_csv(sys.argv[1])

X = df[["x1", "x2", "x3"]].to_numpy()
y = df["y"].to_numpy()

X1 = np.column_stack([np.ones(len(X)), X])
beta0 = np.linalg.lstsq(X1, y, rcond=None)[0]
resid0 = y - X1 @ beta0

def nll(p):
    a, b1, b2, b3, sigma, nu = p
    if sigma <= 0 or nu <= 2:
        return 1e30
    r = y - (a + b1 * X[:, 0] + b2 * X[:, 1] + b3 * X[:, 2])
    return -np.sum(t.logpdf(r, df=nu, loc=0.0, scale=sigma))

x0 = np.array([beta0[0], beta0[1], beta0[2], beta0[3], np.std(resid0, ddof=1), 5.0])
bounds = [(None, None), (None, None), (None, None), (None, None), (1e-12, None), (2.000001, 200.0)]

res = minimize(nll, x0, method="Powell", bounds=bounds, options={"maxiter": 500000, "xtol": 1e-14, "ftol": 1e-14})

a, b1, b2, b3, sigma, nu = res.x

out = pd.DataFrame({
    "mu": [0.0],
    "sigma": [sigma],
    "nu": [nu],
    "Alpha": [a],
    "B1": [b1],
    "B2": [b2],
    "B3": [b3]
})

out.to_csv(sys.argv[2], index=False)

