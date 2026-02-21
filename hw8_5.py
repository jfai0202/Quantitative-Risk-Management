import os
import numpy as np
import pandas as pd
from scipy.stats import t

base_path = os.path.dirname(__file__)
input_path = os.path.join(base_path, "testfiles", "data", "test7_2.csv")
output_path = os.path.join(base_path, "myout_8.5.csv")

df = pd.read_csv(input_path)
r = df["x1"].values

df_t, mu, sigma = t.fit(r)

alpha = 0.05

t_alpha = t.ppf(alpha, df_t)

pdf_val = t.pdf(t_alpha, df_t)

ES_left = mu - sigma * ((df_t + t_alpha**2) / (df_t - 1)) * (pdf_val / alpha)

ES_absolute = -ES_left
ES_diff = ES_absolute - (-mu)

result = pd.DataFrame({
    "ES Absolute": [ES_absolute],
    "ES Diff from Mean": [ES_diff]
})

result.to_csv(output_path, index=False)

print("8.5 Done")