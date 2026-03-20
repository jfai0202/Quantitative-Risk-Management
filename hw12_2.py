import math
import os
import pandas as pd
import numpy as np

base_path = os.path.dirname(__file__)
data_path = os.path.join(base_path, "testfiles", "data")
input_path = os.path.join(data_path, "test12_1.csv")
output_path = os.path.join(base_path, "myout_12_2.csv")


def american_option_crr(option_type, S, K, T, r, q, sigma, steps=500):
    dt = T / steps
    u = math.exp(sigma * math.sqrt(dt))
    d = 1.0 / u
    p = (math.exp((r - q) * dt) - d) / (u - d)
    disc = math.exp(-r * dt)

    stock = np.array([S * (u ** j) * (d ** (steps - j)) for j in range(steps + 1)])

    if option_type.lower() == "call":
        values = np.maximum(stock - K, 0.0)
    else:
        values = np.maximum(K - stock, 0.0)

    for i in range(steps - 1, -1, -1):
        stock = stock[:i + 1] / d
        hold = disc * (p * values[1:i + 2] + (1.0 - p) * values[:i + 1])

        if option_type.lower() == "call":
            exercise = np.maximum(stock - K, 0.0)
        else:
            exercise = np.maximum(K - stock, 0.0)

        values = np.maximum(hold, exercise)

    return float(values[0])


def calc_greeks(option_type, S, K, T, r, q, sigma):
    value = american_option_crr(option_type, S, K, T, r, q, sigma)

    h_delta = 1e-4
    value_up = american_option_crr(option_type, S + h_delta, K, T, r, q, sigma)
    value_down = american_option_crr(option_type, S - h_delta, K, T, r, q, sigma)
    delta = (value_up - value_down) / (2.0 * h_delta)

    h_gamma = 1.5
    value_up_g = american_option_crr(option_type, S + h_gamma, K, T, r, q, sigma)
    value_down_g = american_option_crr(option_type, S - h_gamma, K, T, r, q, sigma)
    gamma = (value_up_g - 2.0 * value + value_down_g) / (h_gamma ** 2)

    h_vega = 1e-4
    value_up_v = american_option_crr(option_type, S, K, T, r, q, sigma + h_vega)
    value_down_v = american_option_crr(option_type, S, K, T, r, q, sigma - h_vega)
    vega = (value_up_v - value_down_v) / (2.0 * h_vega)

    h_rho = 1e-4
    value_up_r = american_option_crr(option_type, S, K, T, r + h_rho, q, sigma)
    value_down_r = american_option_crr(option_type, S, K, T, r - h_rho, q, sigma)
    rho_r = (value_up_r - value_down_r) / (2.0 * h_rho)

    value_up_q = american_option_crr(option_type, S, K, T, r, q + h_rho, sigma)
    value_down_q = american_option_crr(option_type, S, K, T, r, q - h_rho, sigma)
    rho_q = (value_up_q - value_down_q) / (2.0 * h_rho)

    rho = rho_r + rho_q

    h_theta = 1e-4
    value_up_t = american_option_crr(option_type, S, K, T + h_theta, r, q, sigma)
    value_down_t = american_option_crr(option_type, S, K, max(T - h_theta, 1e-8), r, q, sigma)
    theta = (value_up_t - value_down_t) / (2.0 * h_theta)

    return value, delta, gamma, vega, rho, theta


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

    value, delta, gamma, vega, rho, theta = calc_greeks(
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

print("12.2 Done")