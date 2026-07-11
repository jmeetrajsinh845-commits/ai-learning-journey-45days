import pandas as pd
import joblib
import mlflow

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score,f1_score

from xgboost import XGBClassifier



mlflow.set_experiment(
    "Customer_Churn_MLOps"
)



df = pd.read_csv(
    "../data/churn_data.csv"
)



X = df.drop(
    "churn",
    axis=1
)


y=df["churn"]



X_train,X_test,y_train,y_test=train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)



with mlflow.start_run():


    model=XGBClassifier(

        n_estimators=200,

        learning_rate=0.05,

        max_depth=4

    )


    model.fit(
        X_train,
        y_train
    )



    predictions=model.predict(
        X_test
    )


    accuracy=accuracy_score(
        y_test,
        predictions
    )


    f1=f1_score(
        y_test,
        predictions
    )



    mlflow.log_param(
        "model",
        "XGBoost"
    )


    mlflow.log_metric(
        "accuracy",
        accuracy
    )


    mlflow.log_metric(
        "f1_score",
        f1
    )



    mlflow.sklearn.log_model(
        model,
        "churn_model"
    )



joblib.dump(
    model,
    "../models/model_v1.pkl"
)


print("Model training completed")
