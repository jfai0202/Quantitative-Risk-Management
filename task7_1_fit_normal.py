#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import sys
import pandas as pd

df = pd.read_csv(sys.argv[1])
x = df.iloc[:, 0].dropna()

mu = x.mean()
sigma = x.std(ddof=1)

out = pd.DataFrame({
    "mu": [mu],
    "sigma": [sigma]
})

out.to_csv(sys.argv[2], index=False)


# In[ ]:




