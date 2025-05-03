from flask import Flask, request, jsonify
import requests
import os
from functools import lru_cache
import logging
from typing import Dict, List, Tuple, Optional
import statistics
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address

app = Flask(__name__)
limiter = Limiter(app, key_func=get_remote_address)

# Configuration
ODDS_API_KEY = os.getenv("ODDS_API_KEY", "your_api_key_here")
HISTORICAL_DATA_PATH = "historical_results.csv"

# Official Team Registry (Update seasonally)
TEAM_REGISTRY = {
    "Liga Portugal": [
        "Benfica", "Porto", "Sporting", "Braga", "Vitória SC", 
        "Estoril", "Famalicão", "Gil Vicente", "Casa Pia", "Boavista"
    ],
    "Premier League": [
        "Arsenal", "Chelsea", "Liverpool", "Man City", "Man United",
        "Tottenham", "Newcastle", "Aston Villa", "Brighton", "West Ham"
    ]
}

class ValidatedFootballPredictor:
    def __init__(self):
        self.model_name = "Nyam Nyam v6 (Validated)"
        self._init_league_parameters()
        self.historical_data = self._load_historical_data()
        logging.basicConfig(filename='predictions.log', level=logging.INFO)

    def _init_league_parameters(self):
        """League-specific prediction parameters"""
        self.league_params = {
            "Liga Portugal": {
                "tiers": {
                    "Tier 1": ["Benfica", "Porto", "Sporting"],
                    "Tier 2": ["Braga", "Vitória SC"],
                    "Tier 3": ["Estoril", "Famalicão", "Gil Vicente"]
                },
                "goal_expectations": { ... }  # Keep your existing config
            },
            "Premier League": { ... }
        }

    def validate_matchup(self, home_team: str, away_team: str, league: str) -> Tuple[bool, str]:
        """Strict team validation"""
        if home_team == away_team:
            return False, "A team cannot play itself"
        
        if league not in TEAM_REGISTRY:
            return False, f"Unsupported league: {league}"
        
        valid_teams = TEAM_REGISTRY[league]
        errors = []
        
        if home_team not in valid_teams:
            errors.append(f"Invalid home team: {home_team}")
        if away_team not in valid_teams:
            errors.append(f"Invalid away team: {away_team}")
        
        if errors:
            return False, ", ".join(errors)
        
        return True, "Valid matchup"

    def _load_historical_data(self) -> List[Dict]:
        """Load and validate historical data"""
        try:
            import pandas as pd
            df = pd.read_csv(HISTORICAL_DATA_PATH)
            
            # Validate historical records
            valid_matches = []
            for _, row in df.iterrows():
                valid, _ = self.validate_matchup(row["home_team"], row["awayteam"], row["league"])
                if valid:
                    valid_matches.append(row.to_dict())
            
            return valid_matches
        except Exception as e:
            logging.error(f"Historical data error: {str(e)}")
            return []

    @lru_cache(maxsize=32)
    def _get_cached_odds(self, home_team: str, away_team: str, league: str) -> Optional[Tuple[float, float, float]]:
        """Fetch odds only for validated matchups"""
        valid, msg = self.validate_matchup(home_team, away_team, league)
        if not valid:
            logging.warning(f"Odds fetch blocked: {msg}")
            return None
        
        try:
            sport_key = "soccer_portugal_primeira_liga" if league == "Liga Portugal" else "soccer_epl"
            response = requests.get(
                f"https://api.the-odds-api.com/v4/sports/{sport_key}/odds",
                params={"apiKey": ODDS_API_KEY, "regions": "eu", "markets": "h2h"},
                timeout=10
            )
            
            for match in response.json():
                if (home_team.lower() == match["home_team"].lower() and 
                    away_team.lower() == match["away_team"].lower()):
                    outcomes = match["bookmakers"][0]["markets"][0]["outcomes"]
                    return (
                        next(o["price"] for o in outcomes if o["name"] == match["home_team"]),
                        next(o["price"] for o in outcomes if o["name"] == "Draw"),
                        next(o["price"] for o in outcomes if o["name"] == match["away_team"])
                    )
        except Exception as e:
            logging.error(f"Odds API error: {str(e)}")
        return None

    def predict(self, home_team: str, away_team: str, league: str) -> Dict:
        """Core prediction with full validation"""
        # 1. Validate input
        valid, msg = self.validate_matchup(home_team, away_team, league)
        if not valid:
            return {"error": msg, "valid": False}
        
        # 2. Get odds
        odds = self._get_cached_odds(home_team, away_team, league)
        if not odds:
            return {"error": "Could not fetch odds for valid matchup", "valid": False}
        
        # ... [Rest of your existing prediction logic] ...
        
        return {
            "match": f"{home_team} vs {away_team}",
            "valid": True,
            "prediction": prediction,
            "confidence": confidence,
            "data_sources": {
                "teams_validated": True,
                "league": league,
                "team_tiers": {
                    "home": self._get_team_tier(home_team, league),
                    "away": self._get_team_tier(away_team, league)
                }
            }
        }

# Initialize predictor
predictor = ValidatedFootballPredictor()

@app.route("/predict", methods=["POST"])
@limiter.limit("10/minute")
def api_predict():
    data = request.get_json()
    
    # Required fields check
    required = ["home_team", "away_team", "league"]
    if not all(field in data for field in required):
        return jsonify({"error": "Missing required fields", "valid": False}), 400
    
    # Get prediction
    result = predictor.predict(data["home_team"], data["away_team"], data["league"])
    
    # Return error if invalid
    if not result.get("valid", False):
        return jsonify(result), 400
    
    return jsonify(result)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8000)
