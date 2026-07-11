import pandas as pd

import joblib

from sklearn.metrics import classification_report



model=joblib.load(
    "../models/model_v1.pkl"
)



df=pd.read_csv(
    "../data/churn_data.csv"
)



X=df.drop(
    "churn",
    axis=1
)


y=df["churn"]



prediction=model.predict(
    X
)



print(
    classification_report(
        y,
        prediction
    )
)
