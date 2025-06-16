import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import joblib

# Load the sample data
data = pd.read_csv("sample_match_data.csv")

# Define target variable
target = "Result"  # Expected to be 'Home Win', 'Draw', or 'Away Win'

# Fill missing league strength based on mapping
league_strength = {
    'Premier League': 5,
    'Ligue 1': 4,
    'Super League Greece': 3,
    'Kenyan Premier League': 2,
    'Unknown': 1
}
data['LeagueStrength_Home'] = data['League_Home'].map(league_strength).fillna(1)
data['LeagueStrength_Away'] = data['League_Away'].map(league_strength).fillna(1)

# Flag for cross-league match
data['IsCrossLeagueMatch'] = (data['Country_Home'] != data['Country_Away']).astype(int)

# Define usable numeric features
features = [
    "OddsHome", "DrawOdds", "AwayOdds",
    "SofascoreRatingHomeTeam", "SofascoreRatingAwayTeam",
    "NumberofmatchesplayedHometeam", "NumberofmatchesplayedAwayteam",
    "TotalgoalsscoredintheseasonHometeam", "TotalgoalsscoredintheseasonAwayteam",
    "TotalgoalsconcededHomeTeam", "TotalgoalsconcededAwayteam",
    "TotalassistsHometeam", "TotalassistsAwayteam",
    "GoalspergameHometeam", "GoalspergameAwayteam",
    "Goalconversion(Hometeam)", "Goalconversion(Awayteam)",
    "ShotsontargetpergameHometeam", "Shotsontargetpergameawayteam",
    "Bigchancespergamehometeam", "Bigchancespergamehometeamawayteam",
    "BigchancesmissedpergameHometeam", "BigchancesmissedpergameAwayteam",
    "Ballpossessionhometeam", "Ballpossessionawayteam",
    "Accuratepassespergamehometeam", "Accuratepassespergameawayteam",
    "Accuratelongballspergamehometeam", "Accuratelongballspergameawayteam",
    "CleansheetsHometeam", "CleansheetsAwayteam",
    "Goalsconcededpergamehometeam", "Goalsconcededpergameawayteam",
    "InterceptionspergameHometeam", "InterceptionspergameAwayteam",
    "Tacklespergamehometeam", "Tacklespergameawayteam",
    "ClearancespergameHometeam", "Clearancespergameawayteam",
    "PenaltygoalsconcededHometeam", "PenaltygoalsconcededAwayteam",
    "Savespergame",
    "DuelswonpergameHometeam", "DuelswonpergameAwayteam",
    "FoulspergameHometeam", "FoulspergameAwayteam",
    "OffsidespergameHometeam", "OffsidespergameAwayteam",
    "GoalkickspergameHometeam", "GoalkickspergameAwayteam",
    "TotalthrowinsHometeam", "TotalthrowinsAwayteam",
    "TotalyellowcardsawardedHometeam", "TotalyellowcardsawardedAwayteam",
    "TotalRedcardsawardedHometeam", "TotalRedcardsawardedAwayteam",
    "LeaguePositionHomeTeam", "LeaguePositionAwayTeam",
    "TotalPointsHome", "TotalPointsAway",
    "TotalshotspergameHometeam", "TotalshotspergameAwayteam",
    "ShotsofftargetpergameHometeam", "Shotsofftargetpergame",
    "BlockedshotspergameHometeam", "BlockedshotspergameAwayteam",
    "CornerspergameHometeam", "CornerspergameAwayteam",
    "FreekickspergameHometeam", "FreekickspergameAwayteam",
    "HitwoodworkHometeam", "HitwoodworkAwayteam",
    "CounterattacksHometeam", "CounterattacksAwayteam",
    "LeagueStrength_Home", "LeagueStrength_Away",
    "IsCrossLeagueMatch"
]

# Drop rows with missing target
data = data.dropna(subset=[target])

# Encode target labels
label_map = {"Home Win": 0, "Draw": 1, "Away Win": 2}
data[target] = data[target].map(label_map)

# Fill missing numeric feature values with league averages, then global average
for feature in features:
    if feature in data.columns:
        if 'Hometeam' in feature:
            data[feature] = data.groupby('League_Home')[feature].transform(lambda x: x.fillna(x.mean()))
        elif 'Awayteam' in feature:
            data[feature] = data.groupby('League_Away')[feature].transform(lambda x: x.fillna(x.mean()))
        data[feature] = data[feature].fillna(data[feature].mean())

X = data[features]
y = data[target]

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Evaluate
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Model Accuracy: {accuracy:.2f}")
print(classification_report(y_test, y_pred))

# Save model
joblib.dump(model, "football_match_predictor.pkl")
print("Model saved as 'football_match_predictor.pkl'")
