"""
Day 8 — Hyperparameter Tuning: GridSearchCV vs Optuna
======================================================
Churn prediction project continued from Days 5, 6, 7.
Run this file to reproduce all tuning experiments.
"""

import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings("ignore")

from sklearn.model_selection import (
    GridSearchCV,
    StratifiedKFold,
    cross_val_score,
)
from sklearn.metrics import (
    f1_score,
    roc_auc_score,
    classification_report,
)
from xgboost import XGBClassifier

try:
    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    OPTUNA_AVAILABLE = True
except ImportError:
    OPTUNA_AVAILABLE = False
    print("Optuna not installed. Run: pip install optuna")
    print("GridSearchCV section will still run.\n")


# ─────────────────────────────────────────────
# 1. DATA SETUP
# ─────────────────────────────────────────────

def build_dataset(n: int = 600, seed: int = 42) -> tuple:
    """
    Rebuild the same synthetic churn dataset used in Days 5-7.
    Replace this with your real CSV if you have one:
        df = pd.read_csv('your_data.csv')
        X  = df.drop('churn_risk', axis=1)
        y  = df['churn_risk']
    """
    np.random.seed(seed)
    df = pd.DataFrame({
        "days_inactive":    np.random.randint(0, 180, n),
        "rfm_score":        np.random.uniform(0, 100, n),
        "total_spend":      np.random.uniform(500, 15_000, n),
        "orders":           np.random.randint(1, 50, n),
        "engagement_score": np.random.uniform(0, 10, n),
    })
    # Churn label: same rule as Day 4 feature engineering
    df["churn_risk"] = (
        (df["days_inactive"] > 60) | (df["rfm_score"] < 25)
    ).astype(int)

    X = df.drop("churn_risk", axis=1)
    y = df["churn_risk"]
    return X, y


# ─────────────────────────────────────────────
# 2. BASELINE (no tuning)
# ─────────────────────────────────────────────

def run_baseline(X, y, spw: float, cv) -> dict:
    print("\n" + "="*55)
    print("STEP 1 — Baseline XGBoost (default params)")
    print("="*55)

    model = XGBClassifier(
        scale_pos_weight=spw,
        eval_metric="logloss",
        random_state=42,
        verbosity=0,
    )
    f1  = cross_val_score(model, X, y, cv=cv, scoring="f1").mean()
    auc = cross_val_score(model, X, y, cv=cv, scoring="roc_auc").mean()

    print(f"  F1  (CV mean) : {f1:.4f}")
    print(f"  AUC (CV mean) : {auc:.4f}")
    return {"F1": round(f1, 4), "AUC": round(auc, 4)}


# ─────────────────────────────────────────────
# 3. GRIDSEARCHCV
# ─────────────────────────────────────────────

def run_grid_search(X, y, spw: float, cv) -> GridSearchCV:
    print("\n" + "="*55)
    print("STEP 2 — GridSearchCV (exhaustive search)")
    print("="*55)

    param_grid = {
        "n_estimators":     [100, 200, 300],
        "learning_rate":    [0.05, 0.1, 0.2],
        "max_depth":        [3, 4, 6],
        "subsample":        [0.7, 0.9],
        "colsample_bytree": [0.7, 0.9],
    }
    total = (
        len(param_grid["n_estimators"])
        * len(param_grid["learning_rate"])
        * len(param_grid["max_depth"])
        * len(param_grid["subsample"])
        * len(param_grid["colsample_bytree"])
    )
    print(f"  Total combinations : {total}")
    print(f"  CV folds           : 5")
    print(f"  Total training runs: {total * 5}")
    print("  Searching... (this takes a few minutes)\n")

    base = XGBClassifier(
        scale_pos_weight=spw,
        eval_metric="logloss",
        random_state=42,
        verbosity=0,
    )

    gs = GridSearchCV(
        estimator=base,
        param_grid=param_grid,
        cv=cv,
        scoring="f1",
        n_jobs=-1,       # use all CPU cores
        verbose=0,
        refit=True,      # auto-refit best model on full data
    )
    gs.fit(X, y)

    print(f"  Best F1  : {gs.best_score_:.4f}")
    print(f"  Best params:")
    for k, v in gs.best_params_.items():
        print(f"    {k}: {v}")

    return gs


# ─────────────────────────────────────────────
# 4. OPTUNA
# ─────────────────────────────────────────────

def run_optuna(X, y, spw: float, cv, n_trials: int = 100):
    if not OPTUNA_AVAILABLE:
        print("\nSkipping Optuna — not installed.")
        return None

    print("\n" + "="*55)
    print(f"STEP 3 — Optuna (smart search, {n_trials} trials)")
    print("="*55)
    print("  Early trials: random exploration")
    print("  Later trials: focused on best region\n")

    def objective(trial):
        params = {
            "n_estimators":     trial.suggest_int("n_estimators", 100, 500),
            # log=True: denser sampling at small values (where LR matters most)
            "learning_rate":    trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
            "max_depth":        trial.suggest_int("max_depth", 3, 7),
            "subsample":        trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
            "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
            "reg_alpha":        trial.suggest_float("reg_alpha", 0.0, 1.0),
            "reg_lambda":       trial.suggest_float("reg_lambda", 0.5, 2.0),
            "scale_pos_weight": spw,
            "eval_metric":      "logloss",
            "random_state":     42,
            "verbosity":        0,
        }
        model = XGBClassifier(**params)
        scores = cross_val_score(model, X, y, cv=cv, scoring="f1", n_jobs=-1)
        return scores.mean()

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)

    print(f"\n  Best F1  : {study.best_value:.4f}")
    print(f"  Best params:")
    for k, v in study.best_params.items():
        print(f"    {k}: {v}")

    return study


# ─────────────────────────────────────────────
# 5. FINAL MODEL — retrain on full data
# ─────────────────────────────────────────────

def build_final_model(best_params: dict, spw: float, X, y) -> XGBClassifier:
    """
    After tuning, retrain the best model on the FULL training set.
    Never evaluate this on data that was used during CV.
    """
    print("\n" + "="*55)
    print("STEP 4 — Final model (retrained on full dataset)")
    print("="*55)

    params = {**best_params, "scale_pos_weight": spw,
              "eval_metric": "logloss", "random_state": 42, "verbosity": 0}
    model = XGBClassifier(**params)
    model.fit(X, y)

    # Feature importance
    imp = pd.Series(model.feature_importances_, index=X.columns)
    imp = imp.sort_values(ascending=False)
    print("\n  Feature importances:")
    for feat, val in imp.items():
        bar = "█" * int(val * 40)
        print(f"    {feat:<20} {bar} {val:.3f}")

    return model


# ─────────────────────────────────────────────
# 6. COMPARISON SUMMARY
# ─────────────────────────────────────────────

def print_summary(baseline: dict, gs: GridSearchCV, study=None):
    print("\n" + "="*55)
    print("RESULTS SUMMARY")
    print("="*55)

    rows = [
        ("Baseline (no tuning)", baseline["F1"], "—"),
        ("GridSearchCV",         round(gs.best_score_, 4), str(gs.best_params_.get("learning_rate"))),
    ]
    if study is not None:
        rows.append(("Optuna", round(study.best_value, 4),
                     str(round(study.best_params.get("learning_rate", 0), 4))))

    header = f"  {'Method':<22} {'F1 (CV)':>10}  {'Best LR':>10}"
    print(header)
    print("  " + "-" * (len(header) - 2))
    for name, f1, lr in rows:
        print(f"  {name:<22} {f1:>10}  {lr:>10}")

    print("\n  Interpretation:")
    print("  - CV F1 is optimistic. Always validate on held-out test data.")
    print("  - If test F1 << CV F1: suspect overfitting, increase regularization.")
    print("  - Threshold tuning is a SEPARATE step after finding best params.")


# ─────────────────────────────────────────────
# 7. MAIN
# ─────────────────────────────────────────────

if __name__ == "__main__":
    print("\nDay 8 — Hyperparameter Tuning")
    print("Churn Prediction: GridSearchCV vs Optuna")
    print("=" * 55)

    # Data
    X, y = build_dataset(n=600, seed=42)
    neg, pos = (y == 0).sum(), (y == 1).sum()
    spw = neg / pos
    print(f"\nDataset: {len(X)} rows | Churn rate: {pos/(neg+pos)*100:.1f}%")
    print(f"scale_pos_weight: {spw:.2f}  (handles class imbalance)")

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    # Run experiments
    baseline = run_baseline(X, y, spw, cv)
    gs       = run_grid_search(X, y, spw, cv)
    study    = run_optuna(X, y, spw, cv, n_trials=100)

    # Summary
    print_summary(baseline, gs, study)

    # Final model from best method
    if study is not None and study.best_value >= gs.best_score_:
        print("\n  → Using Optuna params for final model")
        best_params = study.best_params
    else:
        print("\n  → Using GridSearchCV params for final model")
        best_params = {k: v for k, v in gs.best_params_.items()
                       if k not in ("scale_pos_weight", "eval_metric",
                                    "random_state", "verbosity")}

    final_model = build_final_model(best_params, spw, X, y)
    print("\nDone. final_model is ready for prediction.")
    print("Next: Day 9 — Cross-validation strategies\n")
