import os
import numpy as np
import pandas as pd

base_path = os.path.dirname(__file__)
data_path = os.path.join(base_path, "testfiles", "data")

factor_path = os.path.join(data_path, "test11_2_factor_returns.csv")
stock_path = os.path.join(data_path, "test11_2_stock_returns.csv")
beta_path = os.path.join(data_path, "test11_2_beta.csv")

# handle possible weight filename variants
weights_path_1 = os.path.join(data_path, "test_11_2_weights.csv")
weights_path_2 = os.path.join(data_path, "test11_2_weights.csv")
weights_path = weights_path_1 if os.path.exists(weights_path_1) else weights_path_2

output_path = os.path.join(base_path, "myout_11_2.csv")


# ----------------------------
# Read data
# ----------------------------
factor_returns = pd.read_csv(factor_path)
stock_returns = pd.read_csv(stock_path)
beta_df = pd.read_csv(beta_path)

# weights file: assume one weight column
weights_df = pd.read_csv(weights_path)
weights = weights_df.iloc[:, 0].values.astype(float)

factor_names = list(factor_returns.columns)          # F1, F2, F3
stock_names = list(stock_returns.columns)            # S1, S2, ...
beta_df = beta_df.set_index(beta_df.columns[0])      # first col = Stock
betas = beta_df.loc[stock_names, factor_names].values

F = factor_returns.values                             # T x K
S = stock_returns.values                              # T x N
T, N = S.shape
K = F.shape[1]

# stock alpha / residual return each period
# alpha_{i,t} = stock_{i,t} - sum_f beta_{i,f} * factor_{f,t}
alpha_returns = S - F @ betas.T


# ----------------------------
# 1) Buy-and-hold portfolio path
# ----------------------------
vals = weights.copy()                                 # initial asset values, total wealth = 1 assumed
factor_contribs = []                                  # T x K
alpha_contribs = []                                   # T
portfolio_returns = []                                # T

for t in range(T):
    current_weights = vals / np.sum(vals)

    # factor contribution at time t
    # c_{f,t} = sum_i w_{i,t-1} * beta_{i,f} * F_{f,t}
    c_f_t = np.sum(current_weights[:, None] * betas * F[t], axis=0)

    # alpha contribution at time t
    c_a_t = np.sum(current_weights * alpha_returns[t])

    # actual portfolio return at time t
    r_p_t = np.sum(current_weights * S[t])

    factor_contribs.append(c_f_t)
    alpha_contribs.append(c_a_t)
    portfolio_returns.append(r_p_t)

    # update asset values for buy-and-hold
    vals = vals * (1.0 + S[t])

factor_contribs = np.array(factor_contribs)           # T x K
alpha_contribs = np.array(alpha_contribs)             # T
portfolio_returns = np.array(portfolio_returns)       # T


# ----------------------------
# 2) Total Return
# ----------------------------
factor_total_return = np.prod(1.0 + F, axis=0) - 1.0
alpha_total_return = np.prod(1.0 + alpha_contribs) - 1.0
portfolio_total_return = np.sum(vals) - 1.0


# ----------------------------
# 3) Return Attribution (Cariño linking)
# ----------------------------
if abs(portfolio_total_return) < 1e-12:
    k_p = 1.0
else:
    k_p = np.log(1.0 + portfolio_total_return) / portfolio_total_return

return_attr_f = np.zeros(K)
return_attr_alpha = 0.0

for t in range(T):
    r_pt = portfolio_returns[t]
    if abs(r_pt) < 1e-12:
        k_t = 1.0
    else:
        k_t = np.log(1.0 + r_pt) / r_pt

    scaler = k_t / k_p
    return_attr_f += factor_contribs[t] * scaler
    return_attr_alpha += alpha_contribs[t] * scaler


# ----------------------------
# 4) Vol Attribution
# RC_k = Cov(c_k, r_p) / sigma_p
# use sample std / sample cov => ddof=1
# ----------------------------
portfolio_vol = np.std(portfolio_returns, ddof=1)

vol_attr_f = np.array([
    np.cov(factor_contribs[:, j], portfolio_returns, ddof=1)[0, 1] / portfolio_vol
    for j in range(K)
])

vol_attr_alpha = np.cov(alpha_contribs, portfolio_returns, ddof=1)[0, 1] / portfolio_vol


# ----------------------------
# 5) Output
# ----------------------------
out_df = pd.DataFrame([
    ["TotalReturn", *factor_total_return, alpha_total_return, portfolio_total_return],
    ["Return Attribution", *return_attr_f, return_attr_alpha, np.sum(return_attr_f) + return_attr_alpha],
    ["Vol Attribution", *vol_attr_f, vol_attr_alpha, np.sum(vol_attr_f) + vol_attr_alpha]
], columns=["Value", *factor_names, "Alpha", "Portfolio"])

out_df.to_csv(output_path, index=False)

print("11.2 Done")