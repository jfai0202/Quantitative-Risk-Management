import pandas as pd
import numpy as np
import os

base_dir = os.path.dirname(os.path.abspath(__file__))
data_dir = os.path.join(base_dir, "testfiles", "data")

input_file = os.path.join(data_dir, "test2.csv")
output_file = os.path.join(data_dir, "my_testout_2.3.csv")

df = pd.read_csv(input_file)
X = df.values
T, N = X.shape


def ew_covariance(data, lam):
    """Exponentially weighted covariance with normalized weights."""
    weights = np.array([(1 - lam) * lam**(T - 1 - t) for t in range(T)])
    weights = weights / weights.sum()

    mean = np.average(data, axis=0, weights=weights)
    centered = data - mean

    return (centered * weights[:, None]).T @ centered


# 1) EW variance (lambda = 0.97)
cov_var = ew_covariance(X, lam=0.97)
ew_var = np.diag(cov_var)
ew_std = np.sqrt(ew_var)

# 2) EW correlation (lambda = 0.94)
cov_corr = ew_covariance(X, lam=0.94)
std_corr = np.sqrt(np.diag(cov_corr))
ew_corr = cov_corr / np.outer(std_corr, std_corr)

# 3) Combine into covariance matrix
ew_cov_final = ew_corr * np.outer(ew_std, ew_std)

result = pd.DataFrame(ew_cov_final, columns=df.columns)

result.to_csv(output_file, index=False)