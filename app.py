import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import joblib

# Load the sample data
data = pd.read_csv("sample_match_data.csv")

# Define target and features
target = "Result"  # Expected to be 'Home Win', 'Draw', or 'Away Win'
features = [
"HomeTeam", "AwayTeam",
"Country",	
"OddsHome",	"DrawOdds","AwayOdds",
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
"Goalsconcededpergamehometeam",	"Goalsconcededpergameawayteam",
"InterceptionspergameHometeam",	"InterceptionspergameAwayteam",
"Tacklespergamehometeam", "Tacklespergameawayteam",
"ClearancespergameHometeam", "Clearancespergameawayteam",
"PenaltygoalsconcededHometeam", "PenaltygoalsconcededAwayteam",
"Savespergame",	"Savespergame",
"DuelswonpergameHometeam",	"DuelswonpergameAwayteam",
"FoulspergameHometeam",	"FoulspergameAwayteam",
"OffsidespergameHometeam",	"OffsidespergameAwayteam",
"GoalkickspergameHometeam",	"GoalkickspergameAwayteam",
"TotalthrowinsHometeam", "TotalthrowinsAwayteam",
"TotalyellowcardsawardedHometeam",	"TotalyellowcardsawardedAwayteam",
"TotalRedcardsawardedHometeam",	"TotalRedcardsawardedAwayteam",
"FormHomeTeam",	"FormAwayTeam",
"LeaguePositionHomeTeam",	"LeaguePositionAwayTeam",
"TotalPointsHome",	"TotalPointsAway",
"TotalshotspergameHometeam",	"TotalshotspergameAwayteam",
"ShotsontargetpergameHometeam",	"ShotsontargetpergameAwayteam",
"ShotsofftargetpergameHometeam", "Shotsofftargetpergame",	
"BlockedshotspergameHometeam",	"BlockedshotspergameAwayteam",
"CornerspergameHometeam", "CornerspergameAwayteam",
"FreekickspergameHometeam",	"FreekickspergameAwayteam",
"HitwoodworkHometeam",	"HitwoodworkAwayteam",
"CounterattacksHometeam",	"CounterattacksAwayteam",
"H2H(Latestooldest)",
]

# Drop rows with missing target
data = data.dropna(subset=[target])

# Convert categorical result to numerical classes
label_map = {"Home Win": 0, "Draw": 1, "Away Win": 2}
data[target] = data[target].map(label_map)

# Fill missing values if any
X = data[features].fillna(0)
y = data[target]

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Evaluate model
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Model Accuracy: {accuracy:.2f}")
print(classification_report(y_test, y_pred))

# Save the model
joblib.dump(model, "football_match_predictor.pkl")
print("Model saved as 'football_match_predictor.pkl'")
