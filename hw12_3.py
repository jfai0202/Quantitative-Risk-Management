import math
import os
import pandas as pd
import numpy as np

base_path = os.path.dirname(__file__)
data_path = os.path.join(base_path, "testfiles", "data")
input_path = os.path.join(data_path, "test12_3.csv")
output_path = os.path.join(base_path, "myout_12_3.csv")


def american_binomial_no_div(option_type, S, K, T, r, sigma, steps):
    dt = T / steps
    u = math.exp(sigma * math.sqrt(dt))
    d = 1.0 / u
    p = (math.exp(r * dt) - d) / (u - d)
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


def american_binomial_discrete_div(option_type, S, K, T, r, sigma, div_times, div_amts, steps=200):
    remaining_divs = [(t, a) for t, a in zip(div_times, div_amts) if 0.0 < t < T]

    if len(remaining_divs) == 0:
        return american_binomial_no_div(option_type, S, K, T, r, sigma, steps)

    first_div_time, first_div_amt = remaining_divs[0]

    steps_before_div = max(1, int(round(steps * first_div_time / T)))
    dt = first_div_time / steps_before_div
    u = math.exp(sigma * math.sqrt(dt))
    d = 1.0 / u
    p = (math.exp(r * dt) - d) / (u - d)
    disc = math.exp(-r * dt)

    stock = np.array([S * (u ** j) * (d ** (steps_before_div - j)) for j in range(steps_before_div + 1)])

    values = np.zeros(steps_before_div + 1)

    later_div_times = [t - first_div_time for t, _ in remaining_divs[1:]]
    later_div_amts = [a for _, a in remaining_divs[1:]]

    for j in range(steps_before_div + 1):
        ex_div_stock = max(stock[j] - first_div_amt, 0.0)
        values[j] = american_binomial_discrete_div(
            option_type,
            ex_div_stock,
            K,
            T - first_div_time,
            r,
            sigma,
            later_div_times,
            later_div_amts,
            steps
        )

    for i in range(steps_before_div - 1, -1, -1):
        stock = stock[:i + 1] / d
        hold = disc * (p * values[1:i + 2] + (1.0 - p) * values[:i + 1])

        if option_type.lower() == "call":
            exercise = np.maximum(stock - K, 0.0)
        else:
            exercise = np.maximum(K - stock, 0.0)

        values = np.maximum(hold, exercise)

    return float(values[0])


def parse_dividend_info(date_str, amt_str, day_per_year):
    if pd.isna(date_str) or str(date_str).strip() == "":
        return [], []

    dates = [float(x.strip()) / day_per_year for x in str(date_str).split(",")]
    amts = [float(x.strip()) for x in str(amt_str).split(",")]

    paired = sorted(zip(dates, amts), key=lambda x: x[0])
    div_times = [x[0] for x in paired]
    div_amts = [x[1] for x in paired]

    return div_times, div_amts


df = pd.read_csv(input_path)
df = df.dropna(subset=["ID"]).copy()

results = []

for _, row in df.iterrows():
    option_type = row["Option Type"]
    S = float(row["Underlying"])
    K = float(row["Strike"])
    T = float(row["DaysToMaturity"]) / float(row["DayPerYear"])
    r = float(row["RiskFreeRate"])
    sigma = float(row["ImpliedVol"])
    day_per_year = float(row["DayPerYear"])

    div_times, div_amts = parse_dividend_info(
        row["DividendDates"],
        row["DividendAmts"],
        day_per_year
    )

    value = american_binomial_discrete_div(
        option_type,
        S,
        K,
        T,
        r,
        sigma,
        div_times,
        div_amts,
        steps=200
    )

    results.append({
        "ID": int(row["ID"]),
        "Value": value
    })

out_df = pd.DataFrame(results)
out_df = out_df[["ID", "Value"]]
out_df.to_csv(output_path, index=False)

print("12.3 Done")