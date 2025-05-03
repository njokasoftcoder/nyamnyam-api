from flask import Flask, request, jsonify
import random
import os

app = Flask(__name__, static_url_path='/static')

class MobileFootballPredictor:
    def __init__(self):
        self.model_name = "Nyam Nyam Confidence Fire Prediction"

    def predict_match(self, home_team, away_team):
        outcome_probs = {
            "home_win": round(random.uniform(0.3, 0.6), 2),
            "draw": round(random.uniform(0.2, 0.4), 2),
            "away_win": round(random.uniform(0.2, 0.5), 2)
        }

        prediction = max(outcome_probs, key=outcome_probs.get)
        compromise = "Draw" if abs(outcome_probs["home_win"] - outcome_probs["away_win"]) < 0.1 else prediction.capitalize()

        score_map = {
            "home_win": f"{random.randint(1, 3)}-{random.randint(0, 1)}",
            "draw": f"{random.randint(0, 2)}-{random.randint(0, 2)}",
            "away_win": f"{random.randint(0, 1)}-{random.randint(1, 3)}"
        }

        total_goals = sum(int(x) for x in score_map[prediction].split("-"))
        goal_market = "Over 2.5" if total_goals > 2 else "Under 2.5"

        confidence_score = max(outcome_probs.values())
        confidence = "High" if confidence_score > 0.55 else "Medium" if confidence_score > 0.45 else "Low"

        return {
            "model_name": self.model_name,
            "category": prediction.replace("_", " ").title(),
            "compromise_prediction": compromise,
            "score_prediction": score_map[prediction],
            "goal_market": goal_market,
            "confidence": confidence,
            "logo_url": request.host_url.rstrip("/") + "/static/logo.png"
        }

predictor = MobileFootballPredictor()

@app.route("/", methods=["GET"])
def home():
    return '''
    <html>
        <head><title>Nyam Nyam Predictor</title></head>
        <body style="text-align:center; font-family:sans-serif;">
            <h1>Nyam Nyam Confidence Fire Prediction 🔥</h1>
            <img src="/static/logo.png" alt="Logo" width="300">
            <p>Send a POST request to <code>/predict</code> with home_team and away_team in JSON.</p>
        </body>
    </html>
    '''

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

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8000, debug=True)  # Allow LAN access

