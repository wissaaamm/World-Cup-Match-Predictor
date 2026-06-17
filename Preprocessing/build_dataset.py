import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler


LAMBDA = 0.005
DEFAULT_REST_DAYS = 30  

results = pd.read_csv("Datasets/results.csv")
elo = pd.read_csv("Datasets/elo_ratings.csv")

results["date"] = pd.to_datetime(results["date"])
elo["date"] = pd.to_datetime(elo["date"])



################################################################


def get_elo(team, date):
    """Get the most recent Elo rating for a team before a given date."""
    past = elo[(elo["team"] == team) & (elo["date"] < date)]
    if past.empty:
        return None
    return past.sort_values("date").iloc[-1]["rating"]


def get_performance_history(team, date, results):
    """Exponential decay weighted sum of past results."""
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
            goals_for     = match["home_score"]
            goals_against = match["away_score"]
        else:
            goals_for     = match["away_score"]
            goals_against = match["home_score"]

        if goals_for > goals_against:
            x = 1
        elif goals_for == goals_against:
            x = 0
        else:
            x = -1

        score += x * np.exp(-LAMBDA * delta_days)

    return score


def get_rest_days(team, date, results):
    """Days since last match. Returns DEFAULT_REST_DAYS if no previous match."""
    past = results[
        ((results["home_team"] == team) | (results["away_team"] == team)) &
        (results["date"] < date)
    ]
    if past.empty:
        return DEFAULT_REST_DAYS
    last_match_date = past["date"].max()
    return (date - last_match_date).days


def get_location(team, row):
    """Returns one-hot encoded location from team A's perspective."""
    if row["neutral"]:
        return [0, 1, 0]  # neutral
    elif row["home_team"] == team:
        return [1, 0, 0]  # home
    else:
        return [0, 0, 1]  # away

###############################################################




rows = []

for (index, row) in results.iterrows():

    date = row["date"]
    home_team = row["home_team"]
    away_team = row["away_team"]

    # get elo of the home and away teams
    home_elo  = get_elo(home_team, date)
    away_elo  = get_elo(away_team, date)


    if home_elo is None or away_elo is None:
        continue


    # calculate the preformance history of both teams using the formula we derived earlier
    home_form = get_performance_history(home_team, date, results)
    away_form = get_performance_history(away_team, date, results)

    # fetch the number of rest days (if no previous match found set it to 4) 
    home_rest = get_rest_days(home_team, date, results)
    away_rest = get_rest_days(away_team, date, results)

    # find the match location
    location = get_location(home_team, row)

    # create a new row with the above info and append it to the dataset

    rows.append({
        "date":          date,
        "home_team":     home_team,
        "away_team":     away_team,
        "home_elo":      home_elo,
        "away_elo":      away_elo,
        "home_form":     home_form,
        "away_form":     away_form,
        "home_rest":     home_rest,
        "away_rest":     away_rest,
        "loc_home":      location[0],
        "loc_neutral":   location[1],
        "loc_away":      location[2],
        "home_goals":    row["home_score"],
        "away_goals":    row["away_score"],
    })

df = pd.DataFrame(rows)



# normalize the elo and subtract them into a new column

scaler_elo  = StandardScaler()
df["home_elo_norm"]  = scaler_elo.fit_transform(df[["home_elo"]].values)
df["away_elo_norm"]  = scaler_elo.transform(df[["away_elo"]].values)
df["elo_diff"]  = df["home_elo_norm"]  - df["away_elo_norm"]


# normalize the team preformance and subtract them

scaler_form = StandardScaler()
df["home_form_norm"] = scaler_form.fit_transform(df[["home_form"]].values)
df["away_form_norm"] = scaler_form.transform(df[["away_form"]].values)
df["form_diff"]  = df["home_form_norm"]  - df["away_form_norm"]


# normalize the date using a log transformation

df["home_rest_log"] = np.log1p(df["home_rest"])  # log1p = log(1 + x)
df["away_rest_log"] = np.log1p(df["away_rest"])
df["rest_diff"] = df["home_rest_log"] - df["away_rest_log"]


# save the dataset 
df.to_csv("Datasets/dataset.csv", index=False)
