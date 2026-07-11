from fastapi import FastAPI

from pydantic import BaseModel

import joblib

import pandas as pd



app = FastAPI(
    title="Customer Churn Prediction API",
    description="Production ML API",
    version="1.0"
)



model = joblib.load(
    "../models/churn_model.pkl"
)



class CustomerData(BaseModel):

    days_inactive:int

    rfm_score:float

    total_spend:float

    orders:int

    engagement_score:float



@app.get("/")
def home():

    return {
        "message":
        "ML Model API Running"
    }




@app.post("/predict")
def predict(
    customer:CustomerData
):


    data = pd.DataFrame(
        [customer.dict()]
    )


    prediction = model.predict(
        data
    )[0]


    probability = model.predict_proba(
        data
    )[0][1]



    return {

        "prediction":
        int(prediction),

        "churn_probability":
        float(probability)

    }
  
