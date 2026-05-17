"""
Optuna Visualization — run AFTER tune.py
Generates plots to understand what the search found.
Requires: optuna, plotly
"""

import numpy as np
import pandas as pd
import optuna
import warnings
warnings.filterwarnings("ignore")
optuna.logging.set_verbosity(optuna.logging.WARNING)

from sklearn.model_selection import StratifiedKFold, cross_val_score
from xgboost import XGBClassifier


def build_dataset(n=600, seed=42):
    np.random.seed(seed)
    df = pd.DataFrame({
        "days_inactive":    np.random.randint(0, 180, n),
        "rfm_score":        np.random.uniform(0, 100, n),
        "total_spend":      np.random.uniform(500, 15_000, n),
        "orders":           np.random.randint(1, 50, n),
        "engagement_score": np.random.uniform(0, 10, n),
    })
    df["churn_risk"] = (
        (df["days_inactive"] > 60) | (df["rfm_score"] < 25)
    ).astype(int)
    return df.drop("churn_risk", axis=1), df["churn_risk"]


if __name__ == "__main__":
    X, y = build_dataset()
    spw = (y == 0).sum() / (y == 1).sum()
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    def objective(trial):
        params = {
            "n_estimators":     trial.suggest_int("n_estimators", 100, 500),
            "learning_rate":    trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
            "max_depth":        trial.suggest_int("max_depth", 3, 7),
            "subsample":        trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
            "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
            "reg_alpha":        trial.suggest_float("reg_alpha", 0.0, 1.0),
            "reg_lambda":       trial.suggest_float("reg_lambda", 0.5, 2.0),
            "scale_pos_weight": spw,
            "eval_metric": "logloss",
            "random_state": 42,
            "verbosity": 0,
        }
        model = XGBClassifier(**params)
        return cross_val_score(model, X, y, cv=cv, scoring="f1", n_jobs=-1).mean()

    print("Running 100-trial Optuna study for visualization...")
    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=100, show_progress_bar=True)

    print(f"\nBest F1: {study.best_value:.4f}")

    # Plot 1: How F1 improved over trials
    fig1 = optuna.visualization.plot_optimization_history(study)
    fig1.update_layout(title="F1 improvement over trials")
    fig1.show()
    fig1.write_html("optuna_optimization_history.html")
    print("Saved: optuna_optimization_history.html")

    # Plot 2: Which params mattered most
    fig2 = optuna.visualization.plot_param_importances(study)
    fig2.update_layout(title="Which hyperparameter moved F1 the most?")
    fig2.show()
    fig2.write_html("optuna_param_importances.html")
    print("Saved: optuna_param_importances.html")

    # Plot 3: Contour — interaction between top 2 params
    fig3 = optuna.visualization.plot_contour(
        study, params=["learning_rate", "max_depth"]
    )
    fig3.update_layout(title="F1 landscape: learning_rate vs max_depth")
    fig3.show()
    fig3.write_html("optuna_contour.html")
    print("Saved: optuna_contour.html")

    print("\nInsight: Use param importances to narrow your next grid search.")
    print("High-importance params → tune carefully.")
    print("Low-importance params → fix at a sensible default, save compute.")
