import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report
import joblib

# Load your dataset
file_path = 'match_data.csv'  # Replace with your dataset path
data = pd.read_csv(file_path)

# --- Feature Engineering ---

def calculate_form_score(form_string):
    form_map = {'W': 3, 'D': 1, 'L': 0}
    return sum([form_map.get(ch.upper(), 0) for ch in str(form_string)])

def parse_h2h(h2h_string):
    wins, draws, losses = 0, 0, 0
    matches = str(h2h_string).split(',')
    for match in matches:
        if '(' not in match or ')' not in match:
            continue
        result_part = match.split('(')[-1].strip(')')
        if ':' not in result_part:
            continue
        try:
            home_goals, away_goals = map(int, result_part.split(':'))
            if home_goals > away_goals:
                wins += 1
            elif home_goals == away_goals:
                draws += 1
            else:
                losses += 1
        except:
            continue
    return pd.Series([wins, draws, losses])

# Apply transformations
data['FormHomeTeamScore'] = data['FormHomeTeam'].apply(calculate_form_score)
data['FormAwayTeamScore'] = data['FormAwayTeam'].apply(calculate_form_score)
data[['H2H_HomeWins', 'H2H_Draws', 'H2H_Losses']] = data['H2H(Latestooldest)'].apply(parse_h2h)

# Drop original non-numeric features not used in modeling
data.drop(columns=['FormHomeTeam', 'FormAwayTeam', 'H2H(Latestooldest)'], inplace=True)

# Encode the target variable
y = data['MatchOutcome']
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

# Define features
feature_columns = [col for col in data.columns if col != 'MatchOutcome']
X = data[feature_columns]

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42)

# Train model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Evaluate
y_pred = model.predict(X_test)
print(classification_report(y_test, y_pred, target_names=label_encoder.classes_))

# Save model and label encoder
joblib.dump(model, 'match_outcome_model.pkl')
joblib.dump(label_encoder, 'label_encoder.pkl')
print("✅ Model and label encoder saved.")
