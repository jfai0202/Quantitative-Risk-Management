import os
import numpy as np
import pandas as pd
from scipy.optimize import minimize

base_path = os.path.dirname(__file__)
data_path = os.path.join(base_path, "testfiles", "data")
input_path = os.path.join(data_path, "test5_2.csv")
output_path = os.path.join(base_path, "myout_10_1.csv")


def risk_parity_weights(cov):
    n = cov.shape[0]
    b = np.ones(n) / n

    # Objective for equal risk contribution under long-only setting
    # Minimize: 1/2 x'Σx - sum(b_i * log(x_i))
    def objective(x):
        return 0.5 * x @ cov @ x - np.sum(b * np.log(x))

    def gradient(x):
        return cov @ x - b / x

    x0 = np.ones(n)
    bounds = [(1e-12, None)] * n

    result = minimize(
        objective,
        x0,
        jac=gradient,
        method="L-BFGS-B",
        bounds=bounds
    )

    x = result.x
    w = x / np.sum(x)
    return w


df = pd.read_csv(input_path)
cov = df.values

weights = risk_parity_weights(cov)

out_df = pd.DataFrame({"W": weights})
out_df.to_csv(output_path, index=False)

print("10.1 Done")