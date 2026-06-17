import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

elo = pd.read_csv("Datasets/elo_ratings.csv")

# normalize elo ratings

scaler = StandardScaler()
elo["elo_normalized"] = scaler.fit_transform(elo[["rating"]])

elo.to_csv("elo_ratings.csv", index = False)
