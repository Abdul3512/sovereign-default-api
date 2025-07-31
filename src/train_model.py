# src/train_model.py

import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import joblib

# ✅ Fix: point to the correct location of the data
DATA_PATH = os.path.join(os.path.dirname(__file__), "sovereign_default_data.csv")
MODEL_PATH = os.path.join(os.path.dirname(__file__), "default_model.pkl")

# ✅ Load dataset
df = pd.read_csv(DATA_PATH)

# ✅ Preview column names to confirm target label exists
print("📄 Columns:", df.columns.tolist())

# ✅ Drop rows with missing values
df = df.dropna()

# ✅ Check if "default" column exists
if "default" not in df.columns:
    raise ValueError("❌ 'default' column not found in the dataset.")

# ✅ Feature matrix and target vector
X = df.drop("default", axis=1)
y = df["default"]

# ✅ Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# ✅ Train model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# ✅ Save model
joblib.dump(model, MODEL_PATH)
print(f"✅ Model saved to {MODEL_PATH}")
