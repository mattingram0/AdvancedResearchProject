import json
import numpy as np

methods = ["Naive1", "Naive2", "NaiveS", "SES", "Holt", "Damped",
           "Holt-Winters", "Comb", "ARIMA", "SARIMA", "Auto", "Theta",
           "TSO", "ES-RNN-S", "ES-RNN-SW", "ES-RNN-D", "ES-RNN-DW",
           "ES-RNN-I", "ES-RNN-IW"]

# OWA
with open('results_48_seasons_owa.txt') as f:
    res48s = json.load(f)

with open('results_48_owa.txt') as f:
    res48 = json.load(f)

for i in range(1, 49):
    for m in methods:
        res48[str(i)][m] = float(np.around(
            np.mean(res48s[str(i)][m]), decimals=3
        ))

with open('results_48_owa.txt', 'w') as f:
    json.dump(res48, f)

# sMAPE
with open('results_48_seasons_smape.txt') as f:
    res48s = json.load(f)

with open('results_48_smape.txt') as f:
    res48 = json.load(f)

for i in range(1, 49):
    for m in methods:
        res48[str(i)][m] = float(np.around(
            np.mean(res48s[str(i)][m]), decimals=3
        ))

with open('results_48_smape.txt', 'w') as f:
    json.dump(res48, f)

# MASE
with open('results_48_seasons_mase.txt') as f:
    res48s = json.load(f)

with open('results_48_mase.txt') as f:
    res48 = json.load(f)

for i in range(1, 49):
    for m in methods:
        res48[str(i)][m] = float(np.around(
            np.mean(res48s[str(i)][m]), decimals=3
        ))

with open('results_48_mase.txt', 'w') as f:
    json.dump(res48, f)