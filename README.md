This is what will be considered in my model

- Elo rating
- Market Value
- Average squad age
- Average squad experience
- Performance history (With emphasis on more recent matches)
- Rest Days
- Match location -> [Home, Neural, Away]

Our dataset has games since the 1950s so we will only consider the games from the year 2000 onward.
This is what the Preprocessing/date_clipping.py file does

Some country names have been changes over the years so Preprocessing/name_restoring.py standardizes the country names

---------------------------------------------------------------------------------------------

Now lets look at how will each variable in our vector will be calculated:

1) Elo rating 

Ratings will be fetched from the Datasets/elo_ratings.csv dataset

2) Performance history (With emphasis on more recent matches)
3) Rest Days

Performance history cill be calculated using the results.csv, for now we will use the function:
let x <- +1 if the team won : 0 if the team drew, -1 if the team lost
Performance history = Sum of: x / (Δ date) to give higher priority to newer games
(We might change the function later)

Rest days will be the date difference of the current date and the last match

4) Match location 
Can easily be looked up from the results.csv table
We Will use one hot encoding, the it will be a 1x3 vector for [Home, Neural, Away]

---------------------------------------------------------------------------------------------

Now for every game lets create a vector:

[
    difference of the normalized elo
    difference of the normalized performance history
    difference in the number of rest days
    One-hot encoded vector will be defined as the following:
        [1, 0, 0]  → team A is at home
        [0, 1, 0]  → neutral
        [0, 0, 1]  → team A is away
]

We will pass this through a neural network where its last layer will be two neurons, the two neurons will be the goals scored for each team