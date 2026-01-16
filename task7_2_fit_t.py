#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import sys
import pandas as pd
from scipy.stats import t

# read data
df = pd.read_csv(sys.argv[1])
x = df.iloc[:, 0].dropna()

# fit t distribution
nu, mu, sigma = t.fit(x)

# output
out = pd.DataFrame({
    "mu": [mu],
    "sigma": [sigma],
    "nu": [nu]
})

out.to_csv(sys.argv[2], index=False)

