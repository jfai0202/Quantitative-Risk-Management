import os
import numpy as np
import pandas as pd
from scipy.stats import norm

base_path = os.path.dirname(__file__)
input_path = os.path.join(base_path, "testfiles", "data", "test7_1.csv")
output_path = os.path.join(base_path, "myout_8.1.csv")

df = pd.read_csv(input_path)
returns = df["x1"].values

mu = np.mean(returns)
sigma = np.std(returns, ddof=1)

alpha = 0.05   # 95% VaR
z = norm.ppf(alpha)

VaR_diff = abs(z) * sigma
VaR_absolute = -(mu + z * sigma)

result = pd.DataFrame({
    "VaR Absolute": [VaR_absolute],
    "VaR Diff from Mean": [VaR_diff]
})

result.to_csv(output_path, index=False)

print("8.1 Done")