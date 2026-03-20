import math
import os
import pandas as pd

base_path = os.path.dirname(__file__)
data_path = os.path.join(base_path, "testfiles", "data")
input_path = os.path.join(data_path, "test12_1.csv")
output_path = os.path.join(base_path, "myout_12_1.csv")


def norm_cdf(x):
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def norm_pdf(x):
    return math.exp(-0.5 * x * x) / math.sqrt(2.0 * math.pi)


def gbsm_with_greeks(option_type, S, K, T, r, q, sigma):
    d1 = (math.log(S / K) + (r - q + 0.5 * sigma * sigma) * T) / (sigma * math.sqrt(T))
    d2 = d1 - sigma * math.sqrt(T)

    gamma = math.exp(-q * T) * norm_pdf(d1) / (S * sigma * math.sqrt(T))
    vega = S * math.exp(-q * T) * norm_pdf(d1) * math.sqrt(T)

    if option_type.lower() == "call":
        value = S * math.exp(-q * T) * norm_cdf(d1) - K * math.exp(-r * T) * norm_cdf(d2)
        delta = math.exp(-q * T) * norm_cdf(d1)
        theta = (
            -S * math.exp(-q * T) * norm_pdf(d1) * sigma / (2.0 * math.sqrt(T))
            - r * K * math.exp(-r * T) * norm_cdf(d2)
            + q * S * math.exp(-q * T) * norm_cdf(d1)
        )
        rho = K * T * math.exp(-r * T) * norm_cdf(d2)
    else:
        value = K * math.exp(-r * T) * norm_cdf(-d2) - S * math.exp(-q * T) * norm_cdf(-d1)
        delta = -math.exp(-q * T) * norm_cdf(-d1)
        theta = (
            -S * math.exp(-q * T) * norm_pdf(d1) * sigma / (2.0 * math.sqrt(T))
            + r * K * math.exp(-r * T) * norm_cdf(-d2)
            - q * S * math.exp(-q * T) * norm_cdf(-d1)
        )
        rho = -K * T * math.exp(-r * T) * norm_cdf(-d2)

    return value, delta, gamma, vega, theta, rho


df = pd.read_csv(input_path)
df = df.dropna(subset=["ID"]).copy()

results = []

for _, row in df.iterrows():
    option_type = row["Option Type"]
    S = float(row["Underlying"])
    K = float(row["Strike"])
    T = float(row["DaysToMaturity"]) / float(row["DayPerYear"])
    r = float(row["RiskFreeRate"])
    q = float(row["DividendRate"])
    sigma = float(row["ImpliedVol"])

    value, delta, gamma, vega, theta, rho = gbsm_with_greeks(
        option_type, S, K, T, r, q, sigma
    )

    results.append({
        "ID": int(row["ID"]),
        "Value": value,
        "Delta": delta,
        "Gamma": gamma,
        "Vega": vega,
        "Rho": rho,
        "Theta": theta
    })

out_df = pd.DataFrame(results)
out_df = out_df[["ID", "Value", "Delta", "Gamma", "Vega", "Rho", "Theta"]]
out_df.to_csv(output_path, index=False)

print("12.1 Done")