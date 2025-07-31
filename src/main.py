from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import numpy as np
import os

app = FastAPI(title="Sovereign Debt Default Predictor API")

# Load the trained model
model_path = os.path.join(os.path.dirname(__file__), "default_model.pkl")
model = joblib.load(model_path)

# Define the input data format using Pydantic
class DefaultInput(BaseModel):
    gdp_growth: float
    inflation_rate: float
    external_debt: float
    foreign_reserves: float
    political_stability: float

# Define a route for prediction
@app.post("/predict")
def predict_default(data: DefaultInput):
    input_array = np.array([
        [
            data.gdp_growth,
            data.inflation_rate,
            data.external_debt,
            data.foreign_reserves,
            data.political_stability
        ]
    ])
    prediction = model.predict(input_array)[0]
    return {"default_risk": bool(prediction)}
