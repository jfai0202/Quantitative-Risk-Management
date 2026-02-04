import pandas as pd
import numpy as np
import os

base_dir = os.path.dirname(os.path.abspath(__file__))
data_dir = os.path.join(base_dir, "testfiles", "data")

input_file = os.path.join(data_dir, "testout_1.4.csv")
output_file = os.path.join(data_dir, "my_testout_3.4.csv")

df_corr = pd.read_csv(input_file)

if df_corr.columns[0].startswith("Unnamed"):
    df_corr = df_corr.drop(columns=df_corr.columns[0])

A = df_corr.apply(pd.to_numeric).values

# Higham nearest PSD for correlation
for _ in range(100):
    w, V = np.linalg.eigh(A)
    w = np.maximum(w, 0)
    A = V @ np.diag(w) @ V.T
    np.fill_diagonal(A, 1.0)

A = np.round(A, 9)

pd.DataFrame(A).to_csv(output_file, index=False, header=False)