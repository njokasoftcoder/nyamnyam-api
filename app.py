from flask import Flask, request, jsonify
import requests
import random
import os
from typing import Optional, Tuple, Dict, List
import statistics

app = Flask(__name__, static_url_path='/static')

# Configuration
ODDS_API_KEY = os.getenv("ODDS_API_KEY", "4ec57c87c864060fa3194d40366e2bc8")
HISTORICAL_DATA_PATH = "historical_results.csv"

class RobustFootballPredictor:
    def __init__(self):
        self.model_name = "Nyam Nyam Confidence Fire Prediction v3"
        self.league_map = {
            "Liga Portugal": "soccer_portugal_primeira_liga",
            "Premier League": "soccer_epl"
        }
        
        # League-specific parameters
        self.league_params = {
            "Liga Portugal": {
                "strong_teams": ["Benfica", "Porto", "Sporting"],
                "draw_adjustment": 0.85,  # Reduce draw probability
                "home_advantage": 1.1     # Slight boost for home teams
            },
            "Premier League": {
                "strong_teams": ["Man City", "Liverpool", "Arsenal"],
                "draw_adjustment": 1.0,
                "home_advantage": 1.15
            }
        }
        
        # Load historical data for validation
        self.historical_data = self._load_historical_data()

    def _load_historical_data(self) -> Dict[str, List]:
        """Load historical match results for validation"""
        try:
            import pandas as pd
            df = pd.read_csv(HISTORICAL_DATA_PATH)
            return df.to_dict('records')
        except:
            return []

    def _validate_against_history(self, home_team: str, away_team: str, league: str) -> Dict:
        """Check historical H2H and team performance"""
        matches = [m for m in self.historical_data 
                  if m['league'] == league
                  and ((m['home_team'] == home_team and m['away_team'] == away_team)
                      or (m['home_team'] == away_team and m['away_team'] == home_team))]
        
        if not matches:
            return {}
            
        home_wins = sum(1 for m in matches if m['home_team'] == home_team and m['result'] == 'H')
        away_wins = sum(1 for m in matches if m['away_team'] == away_team and m['result'] == 'A')
        draws = len(matches) - home_wins - away_wins
        
        return {
            "home_win_pct": home_wins / len(matches),
            "away_win_pct": away_wins / len(matches),
            "draw_pct": draws / len(matches)
        }

    def fetch_odds(self, home_team: str, away_team: str, league: str) -> Optional[Tuple[float, float, float]]:
        """Fetch best available odds with league-specific endpoint"""
        sport_key = self.league_map.get(league, "soccer_epl")
        url = f"https://api.the-odds-api.com/v4/sports/{sport_key}/odds"
        
        try:
            response = requests.get(url, params={
                "regions": "eu",
                "markets": "h2h",
                "apiKey": ODDS_API_KEY
            }, timeout=10)
            response.raise_for_status()
            
            for match in response.json():
                if (home_team.lower() in match["home_team"].lower() and 
                    away_team.lower() in match["away_team"].lower()):
                    
                    # Get best odds across all bookmakers
                    outcomes = []
                    for bookmaker in match["bookmakers"]:
                        outcomes.extend(bookmaker["markets"][0]["outcomes"])
                    
                    home_odds = min(o["price"] for o in outcomes if o["name"] == match["home_team"])
                    draw_odds = min(o["price"] for o in outcomes if o["name"] == "Draw")
                    away_odds = min(o["price"] for o in outcomes if o["name"] == match["away_team"])
                    
                    return home_odds, draw_odds, away_odds
        except Exception as e:
            app.logger.error(f"Odds API Error: {str(e)}")
        return None

    def predict_match(self, home_team: str, away_team: str, odds: Tuple[float, float, float], league: str) -> Dict:
        """Core prediction logic with historical validation"""
        # Convert odds to probabilities
        home_prob = 1 / odds[0]
        draw_prob = 1 / odds[1]
        away_prob = 1 / odds[2]
        overround = home_prob + draw_prob + away_prob
        probs = {
            "home_win": home_prob / overround,
            "draw": draw_prob / overround,
            "away_win": away_prob / overround
        }
        
        # Apply league-specific adjustments
        params = self.league_params.get(league, {})
        if "strong_teams" in params:
            if away_team in params["strong_teams"]:
                probs["away_win"] *= 1.25
                probs["home_win"] *= 0.6
            elif home_team in params["strong_teams"]:
                probs["home_win"] *= 1.25
                probs["away_win"] *= 0.6
        
        probs["draw"] *= params.get("draw_adjustment", 1.0)
        probs["home_win"] *= params.get("home_advantage", 1.0)
        
        # Normalize probabilities after adjustments
        total = sum(probs.values())
        probs = {k: v/total for k, v in probs.items()}
        
        # Historical validation
        history = self._validate_against_history(home_team, away_team, league)
        if history:
            probs = {
                "home_win": (probs["home_win"] + history["home_win_pct"]) / 2,
                "draw": (probs["draw"] + history["draw_pct"]) / 2,
                "away_win": (probs["away_win"] + history["away_win_pct"]) / 2
            }
        
        # Generate scoreline
        def generate_score():
            if probs["home_win"] > 0.65:
                return f"{random.randint(2,3)}-{random.randint(0,1)}"
            elif probs["away_win"] > 0.65:
                return f"{random.randint(0,1)}-{random.randint(2,3)}"
            else:
                return f"{random.randint(0,2)}-{random.randint(0,2)}"
        
        # Confidence calculation
        max_prob = max(probs.values())
        confidence = (
            "High" if max_prob > 0.7 else
            "Medium" if max_prob > 0.55 else
            "Low"
        )
        
        return {
            "match": f"{home_team} vs {away_team}",
            "league": league,
            "probabilities": {k: round(v, 3) for k, v in probs.items()},
            "predicted_outcome": max(probs, key=probs.get).replace("_", " ").title(),
            "confidence": confidence,
            "expected_score": generate_score(),
            "historical_data_used": bool(history),
            "value_bets": self._calculate_value_bets(probs, odds)
        }
    
    def _calculate_value_bets(self, probs: Dict, odds: Tuple) -> List[Dict]:
        """Identify bets with positive expected value"""
        value_threshold = 0.1
        value_bets = []
        
        if (probs["home_win"] * odds[0]) - 1 > value_threshold:
            value_bets.append({
                "market": "1X2",
                "selection": "Home Win",
                "edge": round(probs["home_win"] * odds[0] - 1, 2)
            })
        
        if (probs["away_win"] * odds[2]) - 1 > value_threshold:
            value_bets.append({
                "market": "1X2",
                "selection": "Away Win",
                "edge": round(probs["away_win"] * odds[2] - 1, 2)
            })
            
        return value_bets

# Initialize predictor
predictor = RobustFootballPredictor()

@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()
    
    # Validate input
    required = ["home_team", "away_team", "league"]
    if not all(field in data for field in required):
        return jsonify({"error": f"Missing fields: {required}"}), 400
    
    # Get odds (API or manual)
    odds = None
    if "odds" in data:
        try:
            odds = tuple(map(float, data["odds"]))
            if len(odds) != 3:
                raise ValueError
        except:
            return jsonify({"error": "Invalid odds format. Use [home, draw, away]"}), 400
    else:
        odds = predictor.fetch_odds(data["home_team"], data["away_team"], data["league"])
        if not odds:
            return jsonify({"error": "Odds unavailable"}), 404
    
    # Generate prediction
    try:
        prediction = predictor.predict_match(
            data["home_team"],
            data["away_team"],
            odds,
            data["league"]
        )
        return jsonify(prediction)
    except Exception as e:
        app.logger.error(f"Prediction error: {str(e)}")
        return jsonify({"error": "Prediction failed"}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8000, debug=True)
