import os
import numpy as np
import pandas as pd
from scipy.stats import norm

base_path = os.path.dirname(__file__)
input_path = os.path.join(base_path, "testfiles", "data", "test7_1.csv")
output_path = os.path.join(base_path, "myout_8.4.csv")

df = pd.read_csv(input_path)
r = df["x1"].values

mu = np.mean(r)
sigma = np.std(r, ddof=1)

alpha = 0.05

z = norm.ppf(alpha)

ES_absolute = - (mu - sigma * norm.pdf(z) / alpha)
ES_diff = ES_absolute - (-mu)

result = pd.DataFrame({
    "ES Absolute": [ES_absolute],
    "ES Diff from Mean": [ES_diff]
})

result.to_csv(output_path, index=False)

print("8.4 Done")