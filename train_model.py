import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import joblib

# Load the sample data
data = pd.read_csv("sample_match_data.csv")

# Define target
target = "Result"
label_map = {"Home Win": 0, "Draw": 1, "Away Win": 2}
data[target] = data[target].map(label_map)

# Define features (update this if your column names differ)
features = [col for col in data.columns if col != "Result"]

# Drop missing values
data = data.dropna(subset=[target])
X = data[features].fillna(0)
y = data[target]

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Evaluate the model
y_pred = model.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))

# Save model
joblib.dump(model, "football_match_predictor.pkl")
print("✅ Model saved as 'football_match_predictor.pkl'")
