# PL Match Outcome Predictor using scikit-learn and past match data (2020-2022)

import pandas as pd

# Load the dataset with match statistics
matches = pd.read_csv("matches.csv", index_col=0)

# Preprocessing: Convert necessary columns for machine learning processing
matches["date"] = pd.to_datetime(matches["date"])  # Convert dates to datetime objects
matches["home_away"] = matches["venue"].astype("category").cat.codes  # Convert 'venue' to numeric codes
matches["opponent_code"] = matches["opponent"].astype("category").cat.codes  # Convert opponent names to numeric codes
matches["match_hour"] = matches["time"].str.replace(":.+", "", regex=True).astype(int)  # Extract match hour from time
matches["weekday"] = matches["date"].dt.dayofweek  # Convert date to the day of the week (0=Monday, 6=Sunday)

# Define the target as whether the result was a win (1) or not (0)
matches["target"] = (matches["result"] == "W").astype(int)

# Import RandomForestClassifier from scikit-learn for machine learning
from sklearn.ensemble import RandomForestClassifier
rf = RandomForestClassifier(n_estimators=100, min_samples_split=10, random_state=1)

# Split the data into training (before 2022) and testing (2022 and later) sets
train_data = matches[matches["date"] < '2022-01-01']
test_data = matches[matches["date"] > '2022-01-01']

# Choose the predictors (features) for the model
predictors = ["home_away", "opponent_code", "match_hour", "weekday"]

# Train the model on the training data
rf.fit(train_data[predictors], train_data["target"])

# Make predictions on the test data
predictions = rf.predict(test_data[predictors])

# Calculate the accuracy of the model
from sklearn.metrics import accuracy_score
accuracy = accuracy_score(test_data["target"], predictions)
print(f"Model Accuracy: {accuracy}")

# Analyze the actual vs predicted results
combined_results = pd.DataFrame({
    "actual": test_data["target"],
    "prediction": predictions
})
print(pd.crosstab(index=combined_results["actual"], columns=combined_results["prediction"]))

# Calculate precision of the model
from sklearn.metrics import precision_score
precision = precision_score(test_data["target"], predictions)
print(f"Model Precision: {precision}")

# Function to calculate rolling averages for team performance (e.g., last 3 games)
def rolling_averages(group, columns, new_columns):
    group = group.sort_values("date")
    rolling_stats = group[columns].rolling(3, closed='left').mean()  # Rolling average of last 3 games
    group[new_columns] = rolling_stats
    group = group.dropna(subset=new_columns)  # Remove rows with missing values
    return group

# Apply rolling averages to match statistics for each team
stats_columns = ["gf", "ga", "sh", "sot", "dist", "fk", "pk", "pkatt"]  # Performance metrics
rolling_columns = [f"{col}_rolling" for col in stats_columns]  # New columns for rolling averages
matches_rolling = matches.groupby("team").apply(lambda x: rolling_averages(x, stats_columns, rolling_columns))
matches_rolling = matches_rolling.reset_index(drop=True)  # Reset index after grouping

# Function to make predictions with rolling averages included
def make_predictions(data, predictors):
    train = data[data["date"] < '2022-01-01']
    test = data[data["date"] > '2022-01-01']
    rf.fit(train[predictors], train["target"])
    preds = rf.predict(test[predictors])
    combined = pd.DataFrame({
        "actual": test["target"],
        "prediction": preds
    }, index=test.index)
    precision = precision_score(test["target"], preds)
    return combined, precision

# Make predictions using additional rolling average features
combined, precision = make_predictions(matches_rolling, predictors + rolling_columns)

# Print updated model precision
print(f"Updated Model Precision with Rolling Averages: {precision}")

# Merge additional information (e.g., date, team, opponent) for better analysis
combined = combined.merge(matches_rolling[["date", "team", "opponent", "result"]], left_index=True, right_index=True)
print(combined.head())  # Show combined results with additional information

# Define custom dictionary to map team names to shorter versions (to handle inconsistent naming)
class CustomMappingDict(dict):
    def __missing__(self, key):
        return key  # If a key is missing, return the key itself

# Map full team names to shorter versions
team_mapping = CustomMappingDict({
    "Brighton and Hove Albion": "Brighton",
    "Manchester United": "Manchester Utd",
    "Tottenham Hotspur": "Tottenham",
    "West Ham United": "West Ham",
    "Wolverhampton Wanderers": "Wolves"
})

# Apply team name mapping to the dataset
combined["short_team"] = combined["team"].map(team_mapping)

# Merge predictions for both home and away teams for match outcome analysis
merged_results = combined.merge(combined, left_on=["date", "short_team"], right_on=["date", "opponent"])

# Print final merged results for both teams in each match
print(merged_results.head())
