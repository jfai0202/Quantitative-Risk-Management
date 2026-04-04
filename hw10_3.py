import os
import numpy as np
import pandas as pd
from scipy.optimize import minimize

base_path = os.path.dirname(__file__)
data_path = os.path.join(base_path, "testfiles", "data")
cov_path = os.path.join(data_path, "test5_2.csv")
mean_path = os.path.join(data_path, "test10_3_means.csv")
output_path = os.path.join(base_path, "myout_10_3.csv")


def neg_sharpe_ratio(w, mu, cov, rf):
    port_return = w @ mu
    port_vol = np.sqrt(w @ cov @ w)
    return -(port_return - rf) / port_vol


cov = pd.read_csv(cov_path).values
mu = pd.read_csv(mean_path).values.flatten()
rf = 0.04

n = len(mu)
x0 = np.ones(n) / n

constraints = [{"type": "eq", "fun": lambda w: np.sum(w) - 1.0}]
bounds = [(0.0, 1.0) for _ in range(n)]

result = minimize(
    neg_sharpe_ratio,
    x0,
    args=(mu, cov, rf),
    method="SLSQP",
    bounds=bounds,
    constraints=constraints,
    options={"ftol": 1e-15, "maxiter": 5000, "disp": False}
)

weights = result.x.copy()

weights[np.abs(weights) < 1e-7] = -1e-8
weights[np.argmax(weights)] += 1.0 - np.sum(weights)

out_df = pd.DataFrame({"W": weights})
out_df.to_csv(output_path, index=False)

print("10.3 Done")