import numpy as np
import pandas as pd

df = pd.read_csv("Datasets/dataset.csv")
df = df.drop(columns = [
    'date',
    'home_team',
    'away_team',
    'home_elo',
    'away_elo',
    'home_form',
    'away_form',
    'home_rest',
    'away_rest',
    'home_goals',
    'away_goals',
    'home_elo_norm',
    'away_elo_norm',
    'home_form_norm',
    'away_form_norm',
    'home_rest_log',
    'away_rest_log'
    ], axis = 1)

# i know this is not ideal but its the fastest was i can write it
df['home_goals'] = pd.read_csv("Datasets/dataset.csv")['home_goals']
df['away_goals'] = pd.read_csv("Datasets/dataset.csv")['away_goals']

df.to_csv("final_dataset.csv", index = False)