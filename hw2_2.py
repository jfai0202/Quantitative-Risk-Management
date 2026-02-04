import pandas as pd
import numpy as np
import os

base_dir = os.path.dirname(os.path.abspath(__file__))
data_dir = os.path.join(base_dir, "testfiles", "data")

input_file = os.path.join(data_dir, "test2.csv")
output_file = os.path.join(data_dir, "my_testout_2.2.csv")

df = pd.read_csv(input_file)

lam = 0.94
X = df.values
T = X.shape[0]

# exponentially weighted covariance
weights = np.array([(1 - lam) * lam**(T - 1 - t) for t in range(T)])
weights = weights / weights.sum()

mean = np.average(X, axis=0, weights=weights)
X_centered = X - mean

ew_cov = (X_centered * weights[:, None]).T @ X_centered

# convert covariance to correlation
std = np.sqrt(np.diag(ew_cov))
ew_corr = ew_cov / np.outer(std, std)

result = pd.DataFrame(ew_corr, columns=df.columns)

result.to_csv(output_file, index=False)