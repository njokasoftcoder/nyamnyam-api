import pandas as pd
import json
import requests

# Update this to your CSV file name
csv_file = "MidweekJackpot_Sportpesa.csv"
api_url = "http://127.0.0.1:8000/predict"  # Change to your API address if needed

# List of required features (same as app.py)
required_columns = [
    'OddsHome', 'DrawOdds', 'AwayOdds',
    'SofascoreRatingHomeTeam', 'SofascoreRatingAwayTeam',
    'NumberofmatchesplayedHometeam', 'NumberofmatchesplayedAwayteam',
    'TotalgoalsscoredintheseasonHometeam', 'TotalgoalsscoredintheseasonAwayteam',
    'TotalgoalsconcededHomeTeam', 'TotalgoalsconcededAwayteam',
    'TotalassistsHometeam', 'TotalassistsAwayteam',
    'GoalspergameHometeam', 'GoalspergameAwayteam',
    'Goalconversion(Hometeam)', 'Goalconversion(Awayteam)',
    'ShotsontargetpergameHometeam', 'Shotsontargetpergameawayteam',
    'Bigchancespergamehometeam', 'Bigchancespergamehometeamawayteam',
    'BigchancesmissedpergameHometeam', 'BigchancesmissedpergameAwayteam',
    'Ballpossessionhometeam', 'Ballpossessionawayteam',
    'Accuratepassespergamehometeam', 'Accuratepassespergameawayteam',
    'Accuratelongballspergamehometeam', 'Accuratelongballspergameawayteam',
    'CleansheetsHometeam', 'CleansheetsAwayteam',
    'Goalsconcededpergamehometeam', 'Goalsconcededpergameawayteam',
    'InterceptionspergameHometeam', 'InterceptionspergameAwayteam',
    'Tacklespergamehometeam', 'Tacklespergameawayteam',
    'ClearancespergameHometeam', 'Clearancespergameawayteam',
    'PenaltygoalsconcededHometeam', 'PenaltygoalsconcededAwayteam',
    'Savespergame',
    'DuelswonpergameHometeam', 'DuelswonpergameAwayteam',
    'FoulspergameHometeam', 'FoulspergameAwayteam',
    'OffsidespergameHometeam', 'OffsidespergameAwayteam',
    'GoalkickspergameHometeam', 'GoalkickspergameAwayteam',
    'TotalthrowinsHometeam', 'TotalthrowinsAwayteam',
    'TotalyellowcardsawardedHometeam', 'TotalyellowcardsawardedAwayteam',
    'TotalRedcardsawardedHometeam', 'TotalRedcardsawardedAwayteam',
    'LeaguePositionHomeTeam', 'LeaguePositionAwayTeam',
    'TotalPointsHome', 'TotalPointsAway',
    'TotalshotspergameHometeam', 'TotalshotspergameAwayteam',
    'ShotsofftargetpergameHometeam', 'ShotsofftargetpergameAwayteam',
    'BlockedshotspergameHometeam', 'BlockedshotspergameAwayteam',
    'CornerspergameHometeam', 'CornerspergameAwayteam',
    'FreekickspergameHometeam', 'FreekickspergameAwayteam',
    'HitwoodworkHometeam', 'HitwoodworkAwayteam',
    'CounterattacksHometeam', 'CounterattacksAwayteam',
    'H2H_HomeWins', 'H2H_Draws', 'H2H_Losses'
]

# STEP 1: Load your CSV
try:
    df = pd.read_csv(csv_file)
    print(f"✅ Loaded CSV: {csv_file} with {len(df)} records.")
except Exception as e:
    print(f"❌ Failed to read CSV: {e}")
    exit(1)

# STEP 2: Validate and fill missing columns
for col in required_columns:
    if col not in df.columns:
        print(f"⚠️ Missing column: {col} – filling with 0")
        df[col] = 0

# Keep only the expected columns (reorder and drop extra)
df = df[required_columns]

# STEP 3: Convert to JSON and send to Flask API
payload = df.to_dict(orient="records")
print(f"🚀 Sending {len(payload)} matches to the prediction API...")

try:
    response = requests.post(api_url, json=payload)
    if response.status_code == 200:
        predictions = response.json().get("predictions", [])
        print("🎯 Predictions:")
        for i, pred in enumerate(predictions):
            print(f"Match {i+1}: {pred}")
    else:
        print(f"❌ API Error: {response.status_code}")
        print(response.text)
except Exception as e:
    print(f"❌ Failed to call API: {e}")
