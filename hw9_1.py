import os
import numpy as np
import pandas as pd
from scipy.stats import norm, t

base_path = os.path.dirname(__file__)

data_path = os.path.join(base_path, "testfiles", "data")

portfolio_path = os.path.join(data_path, "test9_1_portfolio.csv")
returns_path = os.path.join(data_path, "test9_1_returns.csv")
testout_path = os.path.join(data_path, "testout9_1.csv")
output_path = os.path.join(base_path, "myout_9.1.csv")

portfolio = pd.read_csv(portfolio_path)
returns = pd.read_csv(returns_path)
testout = pd.read_csv(testout_path)

np.random.seed(42)
n_sim = 100000
alpha = 0.95

marginals = {}

for _, row in portfolio.iterrows():
    stock = row["Stock"]
    dist = row["Distribution"]
    r = returns[stock].values

    if dist == "Normal":
        mu, sigma = norm.fit(r)
        marginals[stock] = ("Normal", mu, sigma)
    else:
        df_t, mu, sigma = t.fit(r)
        marginals[stock] = ("T", df_t, mu, sigma)

z_data = []

for stock in portfolio["Stock"]:
    r = returns[stock].values
    params = marginals[stock]

    if params[0] == "Normal":
        _, mu, sigma = params
        u = norm.cdf(r, mu, sigma)
    else:
        _, df_t, mu, sigma = params
        u = t.cdf(r, df_t, loc=mu, scale=sigma)

    u = np.clip(u, 1e-10, 1 - 1e-10)
    z = norm.ppf(u)
    z_data.append(z)

z_data = np.column_stack(z_data)

corr = np.corrcoef(z_data.T)
L = np.linalg.cholesky(corr)

z_sim = np.random.normal(size=(n_sim, len(portfolio)))
z_corr = z_sim @ L.T
u_sim = norm.cdf(z_corr)

sim_returns = []

for i, stock in enumerate(portfolio["Stock"]):
    params = marginals[stock]
    u = np.clip(u_sim[:, i], 1e-10, 1 - 1e-10)

    if params[0] == "Normal":
        _, mu, sigma = params
        r_sim = norm.ppf(u, mu, sigma)
    else:
        _, df_t, mu, sigma = params
        r_sim = t.ppf(u, df_t, loc=mu, scale=sigma)

    sim_returns.append(r_sim)

sim_returns = np.column_stack(sim_returns)

results = []
total_loss = np.zeros(n_sim)
total_value = 0

for i, row in portfolio.iterrows():
    stock = row["Stock"]
    holding = row["Holding"]
    price = row["Starting Price"]

    V0 = holding * price
    sim_price = price * (1 + sim_returns[:, i])
    sim_value = holding * sim_price

    loss = V0 - sim_value

    VaR = np.percentile(loss, 95)
    ES = loss[loss >= VaR].mean()

    results.append([
        stock,
        VaR,
        ES,
        VaR / V0,
        ES / V0
    ])

    total_loss += loss
    total_value += V0

VaR_total = np.percentile(total_loss, 95)
ES_total = total_loss[total_loss >= VaR_total].mean()

results.append([
    "Total",
    VaR_total,
    ES_total,
    VaR_total / total_value,
    ES_total / total_value
])

myout = pd.DataFrame(
    results,
    columns=["Stock", "VaR95", "ES95", "VaR95_Pct", "ES95_Pct"]
)

myout.to_csv(output_path, index=False)

print("9.1 Done")