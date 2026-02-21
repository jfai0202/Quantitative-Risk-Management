import os
import numpy as np
import pandas as pd
from scipy.stats import t

base_path = os.path.dirname(__file__)
input_path = os.path.join(base_path, "testfiles", "data", "test7_2.csv")
output_path = os.path.join(base_path, "myout_8.6.csv")

df = pd.read_csv(input_path)
r = df["x1"].values

df_t, mu, sigma = t.fit(r)

np.random.seed(42)

sim = t.rvs(df_t, loc=mu, scale=sigma, size=100000)

alpha = 0.05

q = np.quantile(sim, alpha)

ES_left = np.mean(sim[sim <= q])

ES_absolute = -ES_left
ES_diff = ES_absolute - (-mu)

result = pd.DataFrame({
    "ES Absolute": [ES_absolute],
    "ES Diff from Mean": [ES_diff]
})

result.to_csv(output_path, index=False)

print("8.6 Done")