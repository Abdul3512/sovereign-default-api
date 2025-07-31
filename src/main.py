from fastapi import FastAPI
import joblib
from pydantic import BaseModel
import numpy as np

app = FastAPI()

class DebtData(BaseModel):
    inflation_rate: float
    gdp_growth: float
    debt_to_gdp: float
    reserves: float
    corruption_index: float

# Load model
model = joblib.load("src/default_model.pkl")

@app.get("/")
def root():
    return {"message": "Welcome to the Sovereign Debt Default Prediction API!"}

@app.post("/predict")
def predict_default(data: DebtData):
    features = np.array([[data.inflation_rate, data.gdp_growth, data.debt_to_gdp, data.reserves, data.corruption_index]])
    prediction = model.predict(features)[0]
    return {"default_risk": int(prediction)}
