import os
import numpy as np
import pandas as pd

base_path = os.path.dirname(__file__)
data_path = os.path.join(base_path, "testfiles", "data")
returns_path = os.path.join(data_path, "test11_1_returns.csv")
weights_path = os.path.join(data_path, "test11_1_weights.csv")
output_path = os.path.join(base_path, "myout_11_1.csv")


returns = pd.read_csv(returns_path)
weights = pd.read_csv(weights_path)["W"].values
asset_names = list(returns.columns)

R = returns.values
T, N = R.shape

# ---------------------------------------------------
# 1) Buy-and-hold portfolio path
# ---------------------------------------------------
vals = weights.astype(float).copy()   # asset values relative to initial total wealth = 1
period_contribs = []                  # c_{i,t} = w_{i,t-1} * r_{i,t}
portfolio_returns = []                # r_{p,t}

for t in range(T):
    current_weights = vals / np.sum(vals)
    c_t = current_weights * R[t]
    r_p_t = np.sum(c_t)

    period_contribs.append(c_t)
    portfolio_returns.append(r_p_t)

    vals = vals * (1.0 + R[t])

period_contribs = np.array(period_contribs)
portfolio_returns = np.array(portfolio_returns)

# ---------------------------------------------------
# 2) Total Return
# ---------------------------------------------------
asset_total_return = np.prod(1.0 + R, axis=0) - 1.0
portfolio_total_return = np.sum(vals) - 1.0

# ---------------------------------------------------
# 3) Return Attribution (Cariño linking)
# ---------------------------------------------------
portfolio_k = np.log(1.0 + portfolio_total_return) / portfolio_total_return

return_attr = np.zeros(N)
for t in range(T):
    r_pt = portfolio_returns[t]
    if abs(r_pt) < 1e-12:
        k_t = 1.0
    else:
        k_t = np.log(1.0 + r_pt) / r_pt
    return_attr += period_contribs[t] * (k_t / portfolio_k)

# ---------------------------------------------------
# 4) Vol Attribution
# RC_i = Cov(c_i, r_p) / sigma_p
# use sample covariance/std => ddof = 1
# ---------------------------------------------------
portfolio_vol = np.std(portfolio_returns, ddof=1)
vol_attr = np.array([
    np.cov(period_contribs[:, i], portfolio_returns, ddof=1)[0, 1] / portfolio_vol
    for i in range(N)
])

# ---------------------------------------------------
# 5) Output
# ---------------------------------------------------
out_df = pd.DataFrame([
    ["TotalReturn", *asset_total_return, portfolio_total_return],
    ["Return Attribution", *return_attr, np.sum(return_attr)],
    ["Vol Attribution", *vol_attr, np.sum(vol_attr)]
], columns=["Value", *asset_names, "Portfolio"])

out_df.to_csv(output_path, index=False)

print("11.1 Done")