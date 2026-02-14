import os
import numpy as np
import pandas as pd

base_path = os.path.dirname(__file__)
input_path = os.path.join(base_path, "testfiles", "data", "test5_1.csv")
output_path = os.path.join(base_path, "myout_5.1.csv")

df = pd.read_csv(input_path)
cov_input = df.values

np.random.seed(42)

sim = np.random.multivariate_normal(
    mean=np.zeros(cov_input.shape[0]),
    cov=cov_input,
    size=100000
)

cov_output = np.cov(sim, rowvar=False)

n = cov_output.shape[0]
cols = [f"x{i+1}" for i in range(n)]

pd.DataFrame(cov_output, columns=cols).to_csv(output_path, index=False)

print("5.1 Done")