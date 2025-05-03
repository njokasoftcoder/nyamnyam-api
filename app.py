from flask import Flask, request, jsonify
import requests
import random
import os

app = Flask(__name__, static_url_path='/static')

# Replace with your Odds API key
ODDS_API_KEY = "4ec57c87c864060fa3194d40366e2bc8"

class MobileFootballPredictor:
    def __init__(self):
        self.model_name = "Nyam Nyam Confidence Fire Prediction"

    def fetch_odds(self, home_team, away_team):
        url = "https://api.the-odds-api.com/v4/sports/soccer_epl/odds"
        params = {
            "regions": "eu",
            "markets": "h2h",
            "apiKey": ODDS_API_KEY
        }

        try:
            response = requests.get(url, params=params)
            if response.status_code != 200:
                return None

            data = response.json()
            for match in data:
                if home_team.lower() in match["home_team"].lower() and away_team.lower() in match["away_team"].lower():
                    odds = match["bookmakers"][0]["markets"][0]["outcomes"]
                    home_odds = next(o["price"] for o in odds if o["name"] == match["home_team"])
                    draw_odds = next(o["price"] for o in odds if o["name"] == "Draw")
                    away_odds = next(o["price"] for o in odds if o["name"] == match["away_team"])
                    return home_odds, draw_odds, away_odds
        except:
            pass
        return None

    def generate_score(self, prediction):
        if prediction == "home_win":
            home = random.randint(2, 4)
            away = random.randint(0, 1)
        elif prediction == "draw":
            home = away = random.randint(0, 2)
        else:  # away_win
            home = random.randint(0, 1)
            away = random.randint(2, 4)
        return f"{home}-{away}"

    def predict_match(self, home_team, away_team, odds):
        home_odds, draw_odds, away_odds = odds

        # Convert odds to implied probabilities
        total_inverse = (1 / home_odds) + (1 / draw_odds) + (1 / away_odds)
        home_prob = round((1 / home_odds) / total_inverse, 2)
        draw_prob = round((1 / draw_odds) / total_inverse, 2)
        away_prob = round((1 / away_odds) / total_inverse, 2)

        outcome_probs = {
            "home_win": home_prob,
            "draw": draw_prob,
            "away_win": away_prob
        }

        prediction = max(outcome_probs, key=outcome_probs.get)
        compromise = "Draw" if abs(home_prob - away_prob) < 0.1 else prediction.replace("_", " ").title()

        score_line = self.generate_score(prediction)
        total_goals = sum(int(x) for x in score_line.split("-"))
        goal_market = "Over 2.5" if total_goals > 2 else "Under 2.5"

        confidence_score = max(outcome_probs.values())
        confidence = "High" if confidence_score > 0.60 else "Medium" if confidence_score > 0.45 else "Low"

        return {
            "model_name": self.model_name,
            "Main Prediction": prediction.replace("_", " ").title(),
            "Compromise Prediction": compromise,
            "Score Line": score_line,
            "Goal Market": goal_market,
            "Confidence Level": confidence,
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
    manual_odds = data.get("odds")  # Optional manual odds

    if not home_team or not away_team:
        return jsonify({"error": "Both 'home_team' and 'away_team' are required."}), 400

    odds = predictor.fetch_odds(home_team, away_team)

    if not odds and manual_odds:
        try:
            odds = tuple(map(float, manual_odds))  # (home, draw, away)
        except:
            return jsonify({"error": "Invalid manual odds format. Use [home, draw, away]."}), 400
    elif not odds:
        return jsonify({"error": "Odds not found via API and no manual odds provided."}), 404

    result = predictor.predict_match(home_team, away_team, odds)
    return jsonify(result)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8000, debug=True)
