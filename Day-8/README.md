# Day 8 — Hyperparameter Tuning: GridSearchCV vs Optuna

> Part of a 45-day Data Science + ML + AI learning journey.
> Continuing the **Customer Churn Prediction** project from Days 5, 6, and 7.

---

## What this project covers

Day 8 tackles one of the most important and most skipped skills in ML:
**systematic hyperparameter tuning**.

Most tutorials show you how to train a model. Almost none show you how to find
the parameters that actually make it good. This project does that — using two
real approaches used in production and Kaggle competitions.

---

## Two strategies compared

### GridSearchCV — exhaustive
Tries every combination you specify. Predictable. Auditable. Slower.

```
param_grid = {
    'learning_rate':  [0.05, 0.1, 0.2],   # 3
    'max_depth':      [3, 4, 6],           # 3
    'n_estimators':   [100, 200, 300],     # 3
    'subsample':      [0.7, 0.9],          # 2
    'colsample_bytree': [0.7, 0.9],        # 2
}
# Total: 108 combinations × 5 CV folds = 540 training runs
```

### Optuna — intelligent
Learns which regions of the parameter space are promising.
Focuses more trials where they matter. Finds better params in fewer trials.

```python
learning_rate = trial.suggest_float('learning_rate', 0.01, 0.3, log=True)
max_depth     = trial.suggest_int('max_depth', 3, 7)
# ... Optuna decides what to try next based on what worked so far
```

---

## Results (on synthetic churn dataset)

| Method             | CV F1  | Notes                            |
|--------------------|--------|----------------------------------|
| Baseline (no tune) | ~0.72  | XGBoost defaults                 |
| GridSearchCV       | ~0.78  | 108 combinations, exhaustive     |
| Optuna (100 trials)| ~0.81  | Bayesian, focuses on best region |

> Note: Results vary by run. Optuna is stochastic by design.

---

## Project structure

```
day8-hyperparameter-tuning/
├── src/
│   ├── tune.py               # Main: baseline + GridSearchCV + Optuna
│   └── visualize_optuna.py   # Optuna plots: history, importances, contour
├── requirements.txt
└── README.md
```

---

## How to run

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Run the main tuning script

```bash
python src/tune.py
```

This will:
- Build the synthetic churn dataset
- Run baseline XGBoost
- Run GridSearchCV (takes 3–10 min depending on hardware)
- Run Optuna (100 trials, ~5–15 min)
- Print a comparison summary
- Show feature importances of the best model

### 3. Run Optuna visualizations (optional, opens browser)

```bash
python src/visualize_optuna.py
```

Generates three interactive HTML plots:
- `optuna_optimization_history.html` — how F1 improved over 100 trials
- `optuna_param_importances.html` — which params actually moved the needle
- `optuna_contour.html` — F1 landscape: learning_rate vs max_depth

---

## Key concepts learned

**Why tune at all?**
Default hyperparameters are designed to "work in general." Your data is specific.
Tuning finds parameters that work *for your problem*.

**GridSearchCV vs Optuna — when to use which**

| Situation                              | Use           |
|----------------------------------------|---------------|
| Small grid (< 50 combinations)         | GridSearchCV  |
| Large search space, many params        | Optuna        |
| Need full reproducibility / auditability | GridSearchCV |
| Continuous params (learning_rate)      | Optuna        |
| Time limited (Kaggle deadline)         | Optuna        |
| Production tuning, real stakes         | Optuna        |

**Why `log=True` for learning_rate?**
`suggest_float('lr', 0.01, 0.3)` → uniform spacing → treats `0.25` vs `0.26` the same as `0.01` vs `0.02`. But these are very different. Log scale puts more trials where they matter: the low end.

**The tuning workflow (production)**
1. Baseline with defaults
2. Coarse GridSearchCV (2–3 params, 3 values each)
3. Fine Optuna search (100 trials, wider ranges)
4. Validate: check CV F1 vs test F1 (large gap = overfit)
5. Threshold tuning — separate step, done after param tuning

**Threshold tuning is NOT the same as hyperparameter tuning**
Finding the best model params and finding the best decision threshold are two separate optimizations. Always do both.

---

## Common mistakes to avoid

| Mistake                                   | What to do instead                          |
|-------------------------------------------|---------------------------------------------|
| Tuning on test data                       | Only CV on train set, evaluate test once    |
| `scoring='accuracy'` on imbalanced data   | Use `scoring='f1'` or `'roc_auc'`          |
| `class_weight='balanced'` with XGBoost    | Use `scale_pos_weight=neg/pos`              |
| Too many params in one grid               | Start with 2–3 most important               |
| Not fixing `random_state`                 | Always fix — makes runs reproducible        |
| Treating CV best score as final score     | Validate on a held-out test set             |

---

## Dataset

Synthetic customer churn dataset (600 rows, 5 features).
Replace `build_dataset()` in `tune.py` with your own CSV to use real data:

```python
df = pd.read_csv('your_data.csv')
X  = df.drop('churn_risk', axis=1)
y  = df['churn_risk']
```

Features used:
- `days_inactive` — how long since last purchase
- `rfm_score` — Recency, Frequency, Monetary composite (built Day 4)
- `total_spend` — lifetime spend
- `orders` — number of orders
- `engagement_score` — app/email engagement

---

## Journey so far

| Day | Topic                    | Key outcome                           |
|-----|--------------------------|---------------------------------------|
| 1   | Problem framing          | Defined churn as a business problem   |
| 2   | EDA                      | Found key patterns in data            |
| 3   | Statistics               | Distributions, hypothesis testing     |
| 4   | Feature engineering      | Built RFM score, domain features      |
| 5   | Logistic Regression      | Baseline model, AUC 99.3%             |
| 6   | Random Forest            | Learned when LR beats RF              |
| 7   | XGBoost                  | SHAP values, scale_pos_weight         |
| 8   | Hyperparameter tuning    | GridSearchCV + Optuna ← **you are here** |
| 9   | Cross-validation         | Coming next                           |

---

## Skills demonstrated

- XGBoost with imbalanced data handling
- GridSearchCV with `StratifiedKFold`, `n_jobs=-1`
- Optuna: `suggest_int`, `suggest_float`, `log=True`, `direction='maximize'`
- Optuna visualization: optimization history, param importances, contour plots
- Production tuning workflow: baseline → coarse → fine → validate
- Feature importance from tuned model

---

*Day 8 of 45 | Customer Churn Prediction Series*
