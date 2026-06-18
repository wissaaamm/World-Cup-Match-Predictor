import joblib
import torch
import pandas as pd
import numpy as np
from Network import Network

# ── load data ─────────────────────────────────────────────────────────────────

results = pd.read_csv("Datasets/results.csv")
elo_df  = pd.read_csv("Datasets/elo_ratings.csv")

results["date"] = pd.to_datetime(results["date"])
elo_df["date"]  = pd.to_datetime(elo_df["date"])

# ── load model and scalers ────────────────────────────────────────────────────

model = Network()
model.load_state_dict(torch.load("goal_predictor.pt", weights_only=True))
model.eval()

scaler_elo  = joblib.load("scaler_elo.pkl")
scaler_form = joblib.load("scaler_form.pkl")

# ── constants ─────────────────────────────────────────────────────────────────

LAMBDA          = 0.005
DEFAULT_REST    = 30
LOCATION_VECTORS = {
    "home":    [1, 0, 0],
    "neutral": [0, 1, 0],
    "away":    [0, 0, 1],
}

# ── helper functions ──────────────────────────────────────────────────────────

def get_elo(team, date):
    past = elo_df[(elo_df["team"] == team) & (elo_df["date"] < date)]
    if past.empty:
        return None
    return past.sort_values("date").iloc[-1]["rating"]


def get_performance_history(team, date):
    past = results[
        ((results["home_team"] == team) | (results["away_team"] == team)) &
        (results["date"] < date)
    ]
    if past.empty:
        return 0.0

    score = 0.0
    for _, match in past.iterrows():
        delta_days = (date - match["date"]).days
        if delta_days <= 0:
            continue
        if match["home_team"] == team:
            goals_for, goals_against = match["home_score"], match["away_score"]
        else:
            goals_for, goals_against = match["away_score"], match["home_score"]

        x = 1 if goals_for > goals_against else (0 if goals_for == goals_against else -1)
        score += x * np.exp(-LAMBDA * delta_days)

    return score


def get_rest_days(team, date):
    past = results[
        ((results["home_team"] == team) | (results["away_team"] == team)) &
        (results["date"] < date)
    ]
    if past.empty:
        return DEFAULT_REST
    return (date - past["date"].max()).days


# ── predict ───────────────────────────────────────────────────────────────────

def predict(team1, team2, venue, date=None):
    if date is None:
        date = pd.Timestamp.today()

    elo1 = get_elo(team1, date)
    elo2 = get_elo(team2, date)

    if elo1 is None:
        print(f"  No Elo data found for '{team1}'")
        return
    if elo2 is None:
        print(f"  No Elo data found for '{team2}'")
        return

    form1 = get_performance_history(team1, date)
    form2 = get_performance_history(team2, date)
    rest1 = get_rest_days(team1, date)
    rest2 = get_rest_days(team2, date)

    elo1_norm  = scaler_elo.transform(np.array([[elo1]]))[0][0]
    elo2_norm  = scaler_elo.transform(np.array([[elo2]]))[0][0]
    elo_diff   = elo1_norm - elo2_norm

    form1_norm = scaler_form.transform(np.array([[form1]]))[0][0]
    form2_norm = scaler_form.transform(np.array([[form2]]))[0][0]
    form_diff  = form1_norm - form2_norm

    rest_diff  = np.log1p(rest1) - np.log1p(rest2)

    feature_vector = LOCATION_VECTORS[venue] + [elo_diff, form_diff, rest_diff]
    x = torch.tensor([feature_vector], dtype=torch.float32)

    with torch.no_grad():
        goals = model(x).squeeze().tolist()

    g1, g2 = goals[0], goals[1]

    print(f"\n  {team1} vs {team2}  |  venue: {venue}")
    print(f"  ─────────────────────────────────────────")
    print(f"  Predicted score: {team1} {round(g1)} - {round(g2)} {team2}")
    print(f"  (raw: {g1:.2f} - {g2:.2f})\n")

    return g1, g2


# ── run ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    team1 = input("Team 1: ").strip()
    team2 = input("Team 2: ").strip()
    venue = input("Venue (home / neutral / away): ").strip().lower()

    if venue not in LOCATION_VECTORS:
        print("  Venue must be: home, neutral, or away")
    else:
        predict(team1, team2, venue)