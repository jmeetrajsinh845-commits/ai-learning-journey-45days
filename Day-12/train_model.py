import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

from xgboost import XGBClassifier

import joblib


print("Loading dataset...")


df = pd.read_csv(
    "../data/churn_data.csv"
)


print(df.head())


X = df.drop(
    "churn",
    axis=1
)


y = df["churn"]



X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)



print("Training model...")


model = XGBClassifier(
    n_estimators=300,
    learning_rate=0.05,
    max_depth=5,
    random_state=42
)


model.fit(
    X_train,
    y_train
)



predictions = model.predict(
    X_test
)


print(
    classification_report(
        y_test,
        predictions
    )
)



joblib.dump(
    model,
    "../models/churn_model.pkl"
)


print(
    "Model saved successfully"
)
