import pandas as pd
import numpy as np
import os

base_dir = os.path.dirname(os.path.abspath(__file__))
data_dir = os.path.join(base_dir, "testfiles", "data")

input_file = os.path.join(data_dir, "testout_1.3.csv")
output_file = os.path.join(data_dir, "my_testout_3.1.csv")

# read covariance safely
df_cov = pd.read_csv(input_file)

if df_cov.columns[0].startswith("Unnamed"):
    df_cov = df_cov.drop(columns=df_cov.columns[0])

cov = df_cov.apply(pd.to_numeric).values


# Higham near PSD algorithm
def near_psd_higham(A, tol=1e-10, max_iter=100):
    n = A.shape[0]
    Y = A.copy()
    deltaS = np.zeros_like(A)

    for _ in range(max_iter):
        R = Y - deltaS

        eigvals, eigvecs = np.linalg.eigh(R)
        eigvals = np.maximum(eigvals, 0)
        X = eigvecs @ np.diag(eigvals) @ eigvecs.T

        deltaS = X - R

        Y = X.copy()
        np.fill_diagonal(Y, np.diag(A))

        # convergence check
        if np.linalg.norm(Y - A, ord="fro") < tol:
            break

    return Y


cov_psd = near_psd_higham(cov)

pd.DataFrame(cov_psd).to_csv(output_file, index=False, header=False)