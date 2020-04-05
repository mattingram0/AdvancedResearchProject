import json
import numpy as np

methods = ["Naive1", "Naive2", "NaiveS", "SES", "Holt", "Damped",
           "Holt-Winters", "Comb", "ARIMA", "SARIMA", "Auto", "Theta",
           "ES RNN"]

with open('results_48_seasons.txt') as f:
    res48s = json.load(f)

with open('results_48.txt') as f:
    res48 = json.load(f)

for i in range(1, 49):
    for m in methods:
        res48[str(i)][m] = float(np.around(
            np.mean(res48s[str(i)][m]), decimals=3
        ))

with open('results_48.txt') as f:
    json.dump(res48, f)
