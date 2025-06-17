import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
from sklearn.preprocessing import LabelEncoder
import joblib

# Load your dataset
data = pd.read_csv("football_data.csv")

# =====================
# 1. CLEAN % STRINGS
# =====================
for col in data.columns:
    if data[col].astype(str).str.contains('%').any():
        data[col] = data[col].astype(str).str.replace('%', '', regex=False)
        data[col] = pd.to_numeric(data[col], errors='coerce') / 100

# =====================
# 2. TARGET DEFINITION
# =====================
target = 'Result'
if target not in data.columns:
    raise KeyError(f"Column '{target}' not found in dataset.")

data = data.dropna(subset=[target])
label_encoder = LabelEncoder()
data[target] = label_encoder.fit_transform(data[target])
joblib.dump(label_encoder, 'label_encoder.pkl')

# =============================
# 3. ENCODE CATEGORICAL FIELDS
# =============================
categorical_columns = data.select_dtypes(include='object').columns.drop(target, errors='ignore')
for col in categorical_columns:
    data[col] = LabelEncoder().fit_transform(data[col].astype(str))

# =============================
# 4. HANDLE NUMERIC MISSING
# =============================
numeric_columns = data.select_dtypes(include='number').columns.drop(target)
for feature in numeric_columns:
    if 'League_Home' in data.columns:
        try:
            data[feature] = data.groupby('League_Home')[feature].transform(lambda x: pd.to_numeric(x, errors='coerce').fillna(x.mean()))
        except Exception as e:
            print(f"Warning: {feature} could not be group-imputed: {e}")
    data[feature] = pd.to_numeric(data[feature], errors='coerce')
    data[feature] = data[feature].fillna(data[feature].mean())

# =============================
# 5. TRAINING
# =============================
X = data.drop(columns=[target])
y = data[target]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
print("\nModel Accuracy:", accuracy_score(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred, target_names=label_encoder.classes_))

# =============================
# 6. SAVE MODEL
# =============================
joblib.dump(model, 'football_model.pkl')
print("✅ Model saved as football_model.pkl")
