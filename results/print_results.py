import pandas as pd
import json

print("*** Average and Standard Deviation of OWA for 48 Hour Forecast ***")
with open('results_1.txt') as file:
    print(pd.DataFrame(json.load(file)).sort_values("Average"))

print("\n *** Average OWA for 1 - 48 Hour Forecast ***")
with open('results_48.txt') as file:
    print(pd.DataFrame(json.load(file)))
