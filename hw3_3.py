import pandas as pd
import numpy as np
import os

base_dir = os.path.dirname(os.path.abspath(__file__))
data_dir = os.path.join(base_dir, "testfiles", "data")

input_file = os.path.join(data_dir, "testout_1.3.csv")
output_file = os.path.join(data_dir, "my_testout_3.3.csv")

df_cov = pd.read_csv(input_file)

if df_cov.columns[0].startswith("Unnamed"):
    df_cov = df_cov.drop(columns=df_cov.columns[0])

A = df_cov.apply(pd.to_numeric).values
diag0 = np.diag(A)

# Higham nearest PSD algorithm
for _ in range(100):
    w, V = np.linalg.eigh(A)
    w = np.maximum(w, 0)
    A = V @ np.diag(w) @ V.T
    np.fill_diagonal(A, diag0)

A = np.round(A, 9)

pd.DataFrame(A).to_csv(output_file, index=False, header=False)