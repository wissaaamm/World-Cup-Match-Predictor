import pandas as pd

former_names = pd.read_csv("Datasets/former_names.csv")
results = pd.read_csv("Datasets/results.csv")

results = results[results['date'] >= "2000-1-1"]

results.to_csv("Datasets/results.csv", index = False)
