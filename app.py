from flask import Flask, request, jsonify
import random
import os

app = Flask(__name__, static_url_path='/static')

class MobileFootballPredictor:
    def __init__(self):
        self.model_name = "Nyam Nyam Confidence Fire Prediction"

    def predict_match(self, home_team, away_team):
        # Simulate scraping and model logic
        outcome_probs = {
            "home_win": round(random.uniform(0.3, 0.6), 2),
            "draw": round(random.uniform(0.2, 0.4), 2),
            "away_win": round(random.uniform(0.2, 0.5), 2)
        }

        # Pick top outcome
        prediction = max(outcome_probs, key=outcome_probs.get)

        # Compromise logic
        compromise = "Draw" if abs(outcome_probs["home_win"] - outcome_probs["away_win"]) < 0.1 else prediction.capitalize()

        # Score simulation
        score_map = {
            "home_win": f"{random.randint(1, 3)}-{random.randint(0, 1)}",
            "draw": f"{random.randint(0, 2)}-{random.randint(0, 2)}",
            "away_win": f"{random.randint(0, 1)}-{random.randint(1, 3)}"
        }

        # Goal market logic
        total_goals = sum(int(x) for x in score_map[prediction].split("-"))
        goal_market = "Over 2.5" if total_goals > 2 else "Under 2.5"

        # Confidence logic
        confidence_score = max(outcome_probs.values())
        if confidence_score > 0.55:
            confidence = "High"
        elif confidence_score > 0.45:
            confidence = "Medium"
        else:
            confidence = "Low"

        return {
            "model_name": self.model_name,
            "category": prediction.replace("_", " ").title(),
            "compromise_prediction": compromise,
            "score_prediction": score_map[prediction],
            "goal_market": goal_market,
            "confidence": confidence
        }

# Initialize predictor
predictor = MobileFootballPredictor()

@app.route("/", methods=["GET"])
def home():
    return "Nyam Nyam Confidence Fire Prediction is 🔥 live."

@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()
    home_team = data.get("home_team")
    away_team = data.get("away_team")

    if not home_team or not away_team:
        return jsonify({"error": "Both 'home_team' and 'away_team' are required."}), 400

    result = predictor.predict_match(home_team, away_team)
    return jsonify(result)

@app.route("/logo", methods=["GET"])
def show_logo():
    return '''
    <html>
        <head><title>Nyam Nyam Logo</title></head>
        <body style="text-align:center; font-family:sans-serif;">
            <h1>Nyam Nyam Confidence Fire Prediction 🔥</h1>
            <img src="/static/logo.png" alt="Nyam Nyam Logo" width="300">
        </body>
    </html>
    '''

# Run server
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
