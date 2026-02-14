import os
import numpy as np
import pandas as pd
from scipy.stats import t

base_path = os.path.dirname(__file__)
input_path = os.path.join(base_path, "testfiles", "data", "test7_2.csv")
output_path = os.path.join(base_path, "myout_8.2.csv")

df = pd.read_csv(input_path)
r = df["x1"].values

# MLE fit t distribution
df_t, mu, sigma = t.fit(r)

alpha = 0.05
z = t.ppf(alpha, df_t)

VaR_diff = abs(z) * sigma
VaR_absolute = -(mu + z * sigma)

result = pd.DataFrame({
    "VaR Absolute": [VaR_absolute],
    "VaR Diff from Mean": [VaR_diff]
})

result.to_csv(output_path, index=False)

print("8.2 Done")