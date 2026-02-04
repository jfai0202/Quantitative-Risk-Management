import pandas as pd
import os

base_dir = os.path.dirname(os.path.abspath(__file__))
data_dir = os.path.join(base_dir, "testfiles", "data")

input_file = os.path.join(data_dir, "test1.csv")
output_file = os.path.join(data_dir, "testout_1.1.csv")

df = pd.read_csv(input_file)

# skip rows with missing values
df_clean = df.dropna()

# compute correlation matrix
result = df_clean.corr()

# save without row index (match reference format)
result.to_csv(output_file, index=False)