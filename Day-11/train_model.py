import pandas as pd
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split


data = pd.read_csv("../data/sample_churn_data.csv")


X = data.drop("churn", axis=1)
y = data["churn"]


X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)


model = XGBClassifier(
    n_estimators=200,
    learning_rate=0.1,
    max_depth=4,
    random_state=42
)


model.fit(X_train,y_train)


print("Model training completed")

