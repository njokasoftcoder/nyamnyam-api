import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
import re

# Load CSV
data = pd.read_csv("match_data.csv")  # <- update filename if needed

# --- Clean column names ---
data.columns = (
    data.columns
    .str.strip()
    .str.replace('[^A-Za-z0-9_]+', '', regex=True)
    .str.lower()
)

# Print to confirm cleaned names (optional)
# print(data.columns.tolist())

# --- Convert percentage-like strings to float ---
for col in data.columns:
    if data[col].dtype == 'object' and data[col].astype(str).str.contains('%').any():
        data[col] = data[col].str.replace('%', '').astype(float)

# --- Define Target ---
target = 'result'
if target not in data.columns:
    raise KeyError(f"❌ Missing expected target column '{target}' after cleaning.")

data = data.dropna(subset=[target])

# --- Define Feature List (Based on your headers) ---
features = [
    'sofascoreratinghometeam', 'sofascoreratingawayteam',
    'goalconversionhometeam', 'goalconversionawayteam',
    'ballpossessionhometeam', 'ballpossessionawayteam',
    'accuratepassespergamehometeam', 'accuratepassespergameawayteam',
    'accuratelongballspergamehometeam', 'accuratelongballspergameawayteam',
    'cleansheetshometeam', 'cleansheetsawayteam',
    'formhometeam', 'formawayteam',
    'totalshotspergamehometeam', 'totalshotspergameawayteam',
    'shotsontargetpergamehometeam', 'shotsontargetpergameawayteam',
    'shotsofftargetpergamehometeam', 'shotsofftargetpergameawayteam',
    'blockedshotspergamehometeam', 'blockedshotspergameawayteam',
    'duelswonpergamehometeam', 'duelswonpergameawayteam'
]

# --- Confirm All Features Are Present ---
missing = [col for col in features if col not in data.columns]
if missing:
    print(f"❌ Missing features: {missing}")
    raise KeyError("Please ensure the above columns exist in the cleaned dataset.")

# --- Fill NA values with mean grouped by league if available ---
if 'league_home' in data.columns:
    for feature in features:
        data[feature] = data.groupby('league_home')[feature].transform(
            lambda x: pd.to_numeric(x, errors='coerce').fillna(x.mean())
        )
else:
    for feature in features:
        data[feature] = pd.to_numeric(data[feature], errors='coerce').fillna(data[feature].mean())

# --- Prepare Train/Test Data ---
X = data[features]
y = data[target]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# --- Train Model ---
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# --- Evaluate Model ---
y_pred = model.predict(X_test)
print("✅ Model Evaluation:")
print(classification_report(y_test, y_pred))
print(f"Accuracy: {accuracy_score(y_test, y_pred):.2f}")
