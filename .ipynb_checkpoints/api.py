from fastapi import FastAPI
from pydantic import BaseModel
import joblib
from pyforest import *


pipe = joblib.load('churn_pipe.pkl')


class Data(BaseModel):
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

@app.post('/predict')
def predict(data: Data):
    input_df = pd.DataFrame([data.dict()])

    prediction = pipe.predict(input_df)[0]
    probability = pipe.predict_proba(input_df)[0][1]

    return{
        'Churn_prediction' : int(prediction),
        'churn_probablity' : float(probability)
    }