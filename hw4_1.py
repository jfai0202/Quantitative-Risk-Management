import pandas as pd
import numpy as np
import os

base_dir = os.path.dirname(os.path.abspath(__file__))
data_dir = os.path.join(base_dir, "testfiles", "data")

input_file = os.path.join(data_dir, "testout_3.1.csv")
output_file = os.path.join(data_dir, "my_testout_4.1.csv")

dfA = pd.read_csv(input_file)

# drop extra index column if it exists
if dfA.columns[0].startswith("Unnamed"):
    dfA = dfA.drop(columns=dfA.columns[0])

# convert to numeric matrix
A = dfA.apply(pd.to_numeric).values

# ----- chol_psd -----
n = A.shape[0]
L = np.zeros_like(A)

for i in range(n):
    temp = A[i, i] - np.sum(L[i, :i] ** 2)

    if temp < 1e-12:
        L[i, i] = 0.0
    else:
        L[i, i] = np.sqrt(temp)

    for j in range(i + 1, n):
        if L[i, i] > 0:
            L[j, i] = (A[j, i] - np.sum(L[j, :i] * L[i, :i])) / L[i, i]
        else:
            L[j, i] = 0.0

L = np.round(L, 9)

pd.DataFrame(L).to_csv(output_file, index=False, header=False)