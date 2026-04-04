import os
import numpy as np
import pandas as pd
from scipy.optimize import minimize

base_path = os.path.dirname(__file__)
data_path = os.path.join(base_path, "testfiles", "data")
input_path = os.path.join(data_path, "test5_2.csv")
output_path = os.path.join(base_path, "myout_10_2.csv")


def risk_budget_weights(cov, b):
    n = cov.shape[0]

    def objective(w):
        port_var = w @ cov @ w
        mrc = cov @ w
        rc = w * mrc
        target = b * port_var
        return np.sum((rc - target) ** 2)

    constraints = [
        {"type": "eq", "fun": lambda w: np.sum(w) - 1.0}
    ]
    bounds = [(1e-12, 1.0) for _ in range(n)]
    x0 = np.ones(n) / n

    result = minimize(
        objective,
        x0,
        method="SLSQP",
        bounds=bounds,
        constraints=constraints,
        options={"ftol": 1e-15, "maxiter": 5000, "disp": False}
    )

    return result.x


df = pd.read_csv(input_path)
cov = df.values

raw_b = np.array([1.0, 1.0, 1.0, 1.0, 0.5])
b = raw_b / np.sum(raw_b)

weights = risk_budget_weights(cov, b)

out_df = pd.DataFrame({"W": weights})
out_df.to_csv(output_path, index=False)

print("10.2 Done")