import os
import numpy as np
import pandas as pd

base_path = os.path.dirname(__file__)
input_path = os.path.join(base_path, "testfiles", "data", "test5_2.csv")
output_path = os.path.join(base_path, "myout_5.2.csv")

df = pd.read_csv(input_path)
cov_input = df.values

eigvals, eigvecs = np.linalg.eigh(cov_input)
eigvals[eigvals < 0] = 0
cov_fixed = eigvecs @ np.diag(eigvals) @ eigvecs.T

np.random.seed(42)

eigvals2, eigvecs2 = np.linalg.eigh(cov_fixed)
cov_sqrt = eigvecs2 @ np.diag(np.sqrt(eigvals2))

z = np.random.normal(size=(100000, cov_fixed.shape[0]))
sim = z @ cov_sqrt.T

cov_output = np.cov(sim, rowvar=False)

n = cov_output.shape[0]
cols = [f"x{i+1}" for i in range(n)]

pd.DataFrame(cov_output, columns=cols).to_csv(output_path, index=False)

print("5.2 Done")