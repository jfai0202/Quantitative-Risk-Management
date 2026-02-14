import os
import numpy as np
import pandas as pd
from scipy.stats import t

base_path = os.path.dirname(__file__)
input_path = os.path.join(base_path, "testfiles", "data", "test7_2.csv")
output_path = os.path.join(base_path, "myout_8.3.csv")

df = pd.read_csv(input_path)
r = df["x1"].values

df_t, mu, sigma = t.fit(r)

np.random.seed(42)

sim = t.rvs(df_t, loc=mu, scale=sigma, size=100000)

alpha = 0.05

VaR_absolute = -np.quantile(sim, alpha)
VaR_diff = VaR_absolute - (-mu)

result = pd.DataFrame({
    "VaR Absolute": [VaR_absolute],
    "VaR Diff from Mean": [VaR_diff]
})

result.to_csv(output_path, index=False)

print("8.3 Done")