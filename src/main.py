from fastapi import FastAPI
from pydantic import BaseModel
import pickle
import numpy as np

app = FastAPI()

# Load model
with open("src/default_model.pkl", "rb") as f:
    model = pickle.load(f)

# Define input format
class InputData(BaseModel):
    gdp: float
    inflation: float
    debt_to_gdp_ratio: float
    unemployment_rate: float

# Prediction endpoint
@app.post("/predict")
def predict(data: InputData):
    features = np.array([[data.gdp, data.inflation, data.debt_to_gdp_ratio, data.unemployment_rate]])
    prediction = model.predict(features)
    return {"default_probability": prediction[0]}
