import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
import warnings

warnings.filterwarnings("ignore")

# --- Load dataset ---
data = pd.read_csv("sample_match_data.csv")

# --- Normalize column names ---
data.columns = [col.strip().lower().replace(" ", "").replace("(", "").replace(")", "").replace(",", "") for col in data.columns]

# --- Convert percentage strings (e.g. "52%") to float (52.0) ---
for col in data.columns:
    if data[col].dtype == 'object':
        if data[col].astype(str).str.contains('%').any():
            data[col] = data[col].astype(str).str.replace('%', '', regex=False).astype(float)

# --- Define target column ---
target = "result"
target = target.lower().strip()

if target not in data.columns:
    raise ValueError(f"Target column '{target}' not found in dataset.")

# --- Drop rows with missing target ---
data = data.dropna(subset=[target])

# --- Encode target if categorical ---
if not pd.api.types.is_numeric_dtype(data[target]):
    data[target] = data[target].astype("category").cat.codes

# --- Get numeric columns (exclude target) ---
numeric_columns = data.select_dtypes(include='number').columns.tolist()
features = [col for col in numeric_columns if col != target]

# --- Fill missing values grouped by league_home if available ---
group_column = "league_home"
if group_column in data.columns:
    for feature in features:
        data[feature] = data.groupby(group_column)[feature].transform(lambda x: x.fillna(x.mean()))
else:
    data[features] = data[features].fillna(data[features].mean())

# --- Train/test split ---
X = data[features]
y = data[target]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# --- Train model ---
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# --- Predict and evaluate ---
y_pred = model.predict(X_test)
print("\n✅ Model Accuracy:", accuracy_score(y_test, y_pred))
print("\n📊 Classification Report:\n", classification_report(y_test, y_pred))

# --- Feature importance ---
importances = model.feature_importances_
feature_scores = pd.Series(importances, index=features).sort_values(ascending=False)
print("\n📈 Top 20 Important Features:\n", feature_scores.head(20))
