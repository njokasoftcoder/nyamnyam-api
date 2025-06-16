from flask import Flask, render_template, request, send_file
import pandas as pd
import joblib
import os
from io import BytesIO

app = Flask(__name__)
model = joblib.load("football_match_predictor.pkl")

# Same feature list used for training
feature_names = [
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
    "CounterattacksHometeam", "CounterattacksAwayteam"
]

@app.route("/", methods=["GET", "POST"])
def index():
    return render_template("upload.html")

@app.route("/predict", methods=["POST"])
def predict():
    file = request.files["file"]
    if not file:
        return "No file uploaded.", 400

    df = pd.read_csv(file)
    if not all(col in df.columns for col in feature_names):
        missing = [col for col in feature_names if col not in df.columns]
        return f"Missing required columns: {', '.join(missing)}", 400

    X = df[feature_names].fillna(0)
    preds = model.predict(X)
    label_map = {0: "Home Win", 1: "Draw", 2: "Away Win"}
    df["Prediction"] = [label_map[p] for p in preds]

    # Return as downloadable CSV
    output = BytesIO()
    df.to_csv(output, index=False)
    output.seek(0)

    return send_file(output, mimetype="text/csv", as_attachment=True, download_name="predictions.csv")

if __name__ == "__main__":
    app.run(debug=True)
