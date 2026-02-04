import pandas as pd
import numpy as np
import os

base_dir = os.path.dirname(os.path.abspath(__file__))
data_dir = os.path.join(base_dir, "testfiles", "data")

input_file = os.path.join(data_dir, "test2.csv")
output_file = os.path.join(data_dir, "my_testout_2.1.csv")

df = pd.read_csv(input_file)

# exponentially weighted covariance with lambda = 0.97
lam = 0.97
X = df.values
T = X.shape[0]

weights = np.array([(1 - lam) * lam**(T - 1 - t) for t in range(T)])
weights = weights / weights.sum()

mean = np.average(X, axis=0, weights=weights)
X_centered = X - mean

ew_cov = (X_centered * weights[:, None]).T @ X_centered

result = pd.DataFrame(ew_cov, columns=df.columns)

result.to_csv(output_file, index=False)