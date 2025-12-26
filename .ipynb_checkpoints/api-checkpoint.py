from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import numpy as np

# Load pipeline
pipe = joblib.load("churn_pipeline.pkl")

# Define request schema
class CustomerData(BaseModel):
    gender: str
    SeniorCitizen: int
    Partner: str
    Dependents: str
    tenure: int
    PhoneService: str
    MultipleLines: str
    InternetService: str
    OnlineSecurity: str
    OnlineBackup: str
    DeviceProtection: str
    TechSupport: str
    StreamingTV: str
    StreamingMovies: str
    Contract: str
    PaperlessBilling: str
    PaymentMethod: str
    MonthlyCharges: float
    TotalCharges: float

app = FastAPI()

@app.post("/predict")
def predict(data: Data):
    # Convert request to DataFrame
    input_df = pd.DataFrame([data.dict()])

    # Predict using pipeline
    prediction = pipe.predict(input_df)[0]
    probability = pipe.predict_proba(input_df)[0][1]

    return {
        "churn_prediction": int(prediction),
        "churn_probability": float(probability)
    }
