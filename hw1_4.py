import pandas as pd
import os

base_dir = os.path.dirname(os.path.abspath(__file__))
data_dir = os.path.join(base_dir, "testfiles", "data")

input_file = os.path.join(data_dir, "test1.csv")
output_file = os.path.join(data_dir, "my_testout_1.4.csv")

df = pd.read_csv(input_file)

# compute correlation using pairwise complete observations
corr_matrix = df.corr()

# save result without overwriting the reference file
corr_matrix.to_csv(output_file)