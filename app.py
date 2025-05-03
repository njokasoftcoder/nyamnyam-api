from flask import Flask, request, jsonify
import requests
import os
from functools import lru_cache
import logging
from typing import Dict, Tuple, Optional
import statistics
import random

app = Flask(__name__)

# Configuration
ODDS_API_KEY = os.getenv("ODDS_API_KEY", "4ec57c87c864060fa3194d40366e2bc8")
HISTORICAL_DATA_PATH = "historical_results.csv"

class EnhancedFootballPredictor:
    def __init__(self):
        self.model_name = "Nyam Nyam v5 (Enhanced)"
        
        # Enhanced league configuration
        self.league_params = {
            "Liga Portugal": {
                "tiers": {
                    "Tier 1": ["Benfica", "Porto", "Sporting"],
                    "Tier 2": ["Braga", "Vitória SC", "Boavista"],
                    "Tier 3": ["Estoril", "Famalicão", "Gil Vicente", "Casa Pia"]
                },
                "goal_expectations": {
                    "Tier 1_home": 2.1,
                    "Tier 1_away": 1.8,
                    "Tier 2_home": 1.5,
                    "Tier 2_away": 1.2,
                    "Tier 3_home": 1.1,
                    "Tier 3_away": 0.9
                },
                "matchup_adjustments": {
                    "Tier 1_vs_Tier 3": {"home_win_boost": 0.15, "away_win_penalty": -0.10},
                    "Tier 2_vs_Tier 3": {"home_win_boost": 0.10, "draw_boost": 0.05}
                },
                "base_draw_prob": 0.25,
                "home_advantage": 0.12,
                "form_weight": 0.3
            }
        }
        
        self.historical_data = self._load_historical_data()
        logging.basicConfig(filename='predictions.log', level=logging.INFO)

    def _load_historical_data(self) -> Dict:
        """Load historical match results"""
        try:
            import pandas as pd
            return pd.read_csv(HISTORICAL_DATA_PATH).to_dict('records')
        except Exception as e:
            logging.error(f"Failed loading historical data: {str(e)}")
            return []

    def _get_team_tier(self, team: str, league: str) -> str:
        """Classify team into strength tier"""
        for tier, teams in self.league_params[league]["tiers"].items():
            if team in teams:
                return tier
        return "Tier 4"  # Default for unclassified teams

    def _get_expected_goals(self, home_team: str, away_team: str, league: str) -> Tuple[float, float]:
        """Get expected goals based on team tiers"""
        params = self.league_params.get(league, {})
        home_tier = self._get_team_tier(home_team, league)
        away_tier = self._get_team_tier(away_team, league)
        
        home_exp = params.get("goal_expectations", {}).get(f"{home_tier}_home", 1.2)
        away_exp = params.get("goal_expectations", {}).get(f"{away_tier}_away", 1.0)
        
        return home_exp, away_exp

    def _apply_matchup_adjustments(self, probs: Dict, home_team: str, away_team: str, league: str) -> Dict:
        """Adjust probabilities based on team tiers matchup"""
        matchup = f"{self._get_team_tier(home_team, league)}_vs_{self._get_team_tier(away_team, league)}"
        adjustments = self.league_params[league].get("matchup_adjustments", {}).get(matchup, {})
        
        probs["home_win"] += adjustments.get("home_win_boost", 0)
        probs["away_win"] += adjustments.get("away_win_penalty", 0)
        probs["draw"] += adjustments.get("draw_boost", 0)
        
        # Normalize to ensure valid probabilities
        total = sum(probs.values())
        return {k: v/total for k, v in probs.items()}

    @lru_cache(maxsize=32)
    def _get_cached_odds(self, home_team: str, away_team: str, league: str) -> Optional[Tuple[float, float, float]]:
        """Fetch odds with caching"""
        try:
            # Use appropriate API endpoint based on league
            sport_key = "soccer_portugal_primeira_liga" if league == "Liga Portugal" else "soccer_epl"
            url = f"https://api.the-odds-api.com/v4/sports/{sport_key}/odds"
            
            response = requests.get(url, params={
                "apiKey": ODDS_API_KEY,
                "regions": "eu",
                "markets": "h2h"
            }, timeout=10)
            
            for match in response.json():
                if (home_team.lower() in match["home_team"].lower() and 
                    away_team.lower() in match["away_team"].lower()):
                    outcomes = match["bookmakers"][0]["markets"][0]["outcomes"]
                    return (
                        next(o["price"] for o in outcomes if o["name"] == match["home_team"]),
                        next(o["price"] for o in outcomes if o["name"] == "Draw"),
                        next(o["price"] for o in outcomes if o["name"] == match["away_team"])
                    )
        except Exception as e:
            logging.error(f"Odds API error: {str(e)}")
        return None

    def _get_historical_stats(self, home_team: str, away_team: str) -> Dict:
        """Calculate historical performance metrics"""
        matches = [m for m in self.historical_data 
                  if (m["home_team"] == home_team and m["away_team"] == away_team) or
                     (m["home_team"] == away_team and m["away_team"] == home_team)]
        
        if not matches:
            return {}
            
        home_wins = sum(1 for m in matches if m["home_team"] == home_team and m["FTR"] == "H")
        away_wins = sum(1 for m in matches if m["away_team"] == away_team and m["FTR"] == "A")
        
        return {
            "home_win_pct": home_wins / len(matches),
            "away_win_pct": away_wins / len(matches),
            "avg_home_goals": statistics.mean(float(m["home_goals"]) for m in matches),
            "avg_away_goals": statistics.mean(float(m["away_goals"]) for m in matches)
        }

    def predict(self, home_team: str, away_team: str, league: str) -> Dict:
        """Generate stable prediction with all data sources"""
        # 1. Get cached odds
        odds = self._get_cached_odds(home_team, away_team, league)
        if not odds:
            return {"error": "Could not fetch odds"}
        
        # 2. Calculate base probabilities from odds
        home_prob = 1 / odds[0]
        draw_prob = 1 / odds[1]
        away_prob = 1 / odds[2]
        probs = {
            "home_win": home_prob,
            "draw": draw_prob,
            "away_win": away_prob
        }
        
        # 3. Apply league base adjustments
        league_settings = self.league_params.get(league, {})
        probs["draw"] = league_settings.get("base_draw_prob", 0.25)
        remaining_prob = 1 - probs["draw"]
        probs["home_win"] = remaining_prob * (0.5 + league_settings.get("home_advantage", 0))
        probs["away_win"] = remaining_prob - probs["home_win"]
        
        # 4. Apply matchup adjustments
        probs = self._apply_matchup_adjustments(probs, home_team, away_team, league)
        
        # 5. Blend with historical data if available
        history = self._get_historical_stats(home_team, away_team)
        if history:
            blend_weight = league_settings.get("form_weight", 0.3)
            probs["home_win"] = (probs["home_win"] * (1 - blend_weight)) + (history["home_win_pct"] * blend_weight)
            probs["away_win"] = (probs["away_win"] * (1 - blend_weight)) + (history["away_win_pct"] * blend_weight)
            probs["draw"] = 1 - probs["home_win"] - probs["away_win"]
        
        # 6. Generate deterministic score
        home_exp, away_exp = self._get_expected_goals(home_team, away_team, league)
        score = f"{round(home_exp)}-{round(away_exp)}"
        
        # 7. Prepare output
        prediction = max(probs, key=probs.get).replace("_", " ").title()
        confidence = "High" if max(probs.values()) > 0.7 else "Medium" if max(probs.values()) > 0.55 else "Low"
        
        logging.info(
            f"Prediction: {home_team} {score} {away_team} | "
            f"Outcome: {prediction} ({confidence}) | "
            f"Probs: H{probs['home_win']:.2f} D{probs['draw']:.2f} A{probs['away_win']:.2f}"
        )
        
        return {
            "match": f"{home_team} vs {away_team}",
            "league": league,
            "prediction": prediction,
            "score": score,
            "confidence": confidence,
            "probabilities": {k: round(v, 3) for k, v in probs.items()},
            "expected_goals": {"home": home_exp, "away": away_exp},
            "data_sources": {
                "odds": odds,
                "historical_stats": history,
                "team_tiers": {
                    "home": self._get_team_tier(home_team, league),
                    "away": self._get_team_tier(away_team, league)
                }
            }
        }

# Initialize predictor
predictor = EnhancedFootballPredictor()

@app.route("/predict", methods=["POST"])
def api_predict():
    data = request.get_json()
    required = ["home_team", "away_team", "league"]
    if not all(field in data for field in required):
        return jsonify({"error": "Missing required fields"}), 400
    
    result = predictor.predict(data["home_team"], data["away_team"], data["league"])
    return jsonify(result)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8000, debug=True)
