import os
import numpy as np
import pandas as pd

base_path = os.path.dirname(__file__)
input_path = os.path.join(base_path, "testfiles", "data", "test5_2.csv")
output_path = os.path.join(base_path, "myout_5.5.csv")

df = pd.read_csv(input_path)
cov_input = df.values

eigvals, eigvecs = np.linalg.eigh(cov_input)

idx = np.argsort(eigvals)[::-1]
eigvals = eigvals[idx]
eigvecs = eigvecs[:, idx]

# cumulative explained variance
explained_ratio = np.cumsum(eigvals) / np.sum(eigvals)

# keep components that explain 99%
k = np.where(explained_ratio >= 0.99)[0][0] + 1

eigvals_k = eigvals[:k]
eigvecs_k = eigvecs[:, :k]

np.random.seed(42)

z = np.random.normal(size=(100000, k))
sim = z @ np.diag(np.sqrt(eigvals_k)) @ eigvecs_k.T

cov_output = np.cov(sim, rowvar=False)

n = cov_output.shape[0]
cols = [f"x{i+1}" for i in range(n)]

pd.DataFrame(cov_output, columns=cols).to_csv(output_path, index=False)

print("5.5 Done")
print("Number of components used:", k)