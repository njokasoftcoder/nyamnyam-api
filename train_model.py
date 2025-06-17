import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
import warnings

warnings.filterwarnings("ignore")

# Load the dataset
data = pd.read_csv("sample_match_data.csv")

# --- Clean column names ---
data.columns = [col.strip().lower().replace(" ", "").replace("(", "").replace(")", "").replace(",", "") for col in data.columns]

# --- Fix percentages: Convert 'ballpossessionhometeam' etc. from "50%" to 50.0 ---
for col in data.columns:
    if data[col].dtype == object and data[col].astype(str).str.contains('%').any():
        data[col] = data[col].str.rstrip('%').astype(float)

# --- Target Column ---
target = "result"
target = target.lower().strip()

if target not in data.columns:
    raise ValueError(f"Target column '{target}' not found in dataset.")

# Drop rows without target
data = data.dropna(subset=[target])

# --- Convert target to categorical if not numeric ---
if not pd.api.types.is_numeric_dtype(data[target]):
    data[target] = data[target].astype("category").cat.codes

# --- Automatically select numeric features (excluding target) ---
numeric_columns = data.select_dtypes(include='number').columns.tolist()
features = [col for col in numeric_columns if col != target]

# --- Fill missing numeric values grouped by league_home (if available) ---
group_column = "league_home"
if group_column in data.columns:
    for feature in features:
        data[feature] = data.groupby(group_column)[feature].transform(lambda x: x.fillna(x.mean()))
else:
    data[features] = data[features].fillna(data[features].mean())

# --- Train/Test Split ---
X = data[features]
y = data[target]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# --- Train Model ---
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# --- Evaluate ---
y_pred = model.predict(X_test)
print("\n✅ Model Accuracy:", accuracy_score(y_test, y_pred))
print("\n📊 Classification Report:\n", classification_report(y_test, y_pred))

# --- Optional: Feature Importance ---
importances = model.feature_importances_
feature_scores = pd.Series(importances, index=features).sort_values(ascending=False)
print("\n📈 Top 20 Important Features:\n", feature_scores.head(20))
