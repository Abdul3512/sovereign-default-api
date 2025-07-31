# src/generate_data.py

import pandas as pd
import numpy as np
import os

# Ensure the src directory exists
os.makedirs("src", exist_ok=True)

np.random.seed(42)

n_samples = 1000

data = {
    "gdp_growth": np.random.uniform(-5, 10, size=n_samples),
    "inflation_rate": np.random.uniform(0, 50, size=n_samples),
    "external_debt": np.random.uniform(10, 150, size=n_samples),
    "foreign_reserves": np.random.uniform(1, 100, size=n_samples),
    "political_stability": np.random.uniform(0, 1, size=n_samples),
}

df = pd.DataFrame(data)

# Add binary target column based on some logic
df["default"] = (
    (df["gdp_growth"] < 1) &
    (df["inflation_rate"] > 20) &
    (df["external_debt"] > 80)
).astype(int)

# Save to src/
df.to_csv("src/sovereign_default_data.csv", index=False)
print("✅ Data saved as 'src/sovereign_default_data.csv'")
