import pandas as pd
import json
import numpy as np

methods = ["Naive1", "Naive2", "NaiveS", "SES", "Holt", "Damped",
		   "Holt-Winters", "Comb", "ARIMA", "SARIMA", "Auto", "Theta"]

with open('results_1.txt') as f:
	res = json.load(f)

# Calculate the averages
for m in methods:
	means = []
	stds = []

	for s in ["Spring", "Summer", "Autumn", "Winter"]:
		pair = res[s][m]
		means.append(pair[0])
		stds.append(pair[1])

	# Just to prevent python complaining when some of the tests haven't
	# finished yet and the results entry for them are empty
	try:
		res["Average"][m] = [
			float(np.around(np.mean(means), decimals=3)),
			float(np.around(np.mean(stds), decimals=3))
		]
	except Warning:
		pass

# Print and results
results = pd.DataFrame(res).sort_values("Average")
print(results)

# Save results with the averages
with open('results_1.txt', 'w') as f:
	json.dump(res, f)

