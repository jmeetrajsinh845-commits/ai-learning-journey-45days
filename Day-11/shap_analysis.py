import shap
import pandas as pd
import matplotlib.pyplot as plt

from xgboost import XGBClassifier


data = pd.read_csv("../data/sample_churn_data.csv")


X = data.drop("churn",axis=1)
y = data["churn"]


model = XGBClassifier()
model.fit(X,y)


explainer = shap.TreeExplainer(model)


shap_values = explainer.shap_values(X)


shap.summary_plot(
    shap_values,
    X
)


plt.savefig(
"../images/shap_summary.png"
)


print("SHAP analysis completed")
