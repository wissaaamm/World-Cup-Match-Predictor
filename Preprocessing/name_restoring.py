import pandas as pd
import numpy as np

former_names = pd.read_csv("Datasets/former_names.csv")

results = pd.read_csv("Datasets/results.csv")

for (index, row) in former_names.iterrows():
    current = row['current']
    former = row['former']

    results['home_team'] = np.where(results['home_team'] == former, current, results['home_team'])
    results['away_team'] = np.where(results['away_team'] == former, current, results['away_team'])
    results['country'] = np.where(results['country'] == former, current, results['country'])

results.to_csv("Datasets/results.csv", index = False)
