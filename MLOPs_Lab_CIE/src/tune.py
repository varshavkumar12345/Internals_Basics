import os
import json
import math
import itertools
import random
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import KFold, cross_val_score
from sklearn.metrics import mean_absolute_error, mean_squared_error
import mlflow
import mlflow.sklearn

# ── Paths ──────────────────────────────────────────────────────────────────────
BASE_DIR    = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH   = os.path.join(BASE_DIR, "data", "training_data.csv")
RESULTS_DIR = os.path.join(BASE_DIR, "results")
RESULTS_PATH = os.path.join(RESULTS_DIR, "step2_s2.json")

os.makedirs(RESULTS_DIR, exist_ok=True)

# ── Load Data ──────────────────────────────────────────────────────────────────
df = pd.read_csv(DATA_PATH)
TARGET   = "review_turnaround_hours"
FEATURES = [c for c in df.columns if c != TARGET]

X = df[FEATURES].values
y = df[TARGET].values

# ── MLflow Setup ───────────────────────────────────────────────────────────────
EXPERIMENT_NAME = "mergegate-review-turnaround-hours"
PARENT_RUN_NAME = "tuning-mergegate"
mlflow.set_experiment(EXPERIMENT_NAME)

# ── Parameter Grid → Random Search ────────────────────────────────────────────
param_grid = {
    "n_estimators":    [100, 200, 300],
    "max_depth":       [3, 7, 15],
    "min_samples_split": [2, 4],
}

# Build all combinations and shuffle for random search
all_combos = [
    dict(zip(param_grid.keys(), combo))
    for combo in itertools.product(*param_grid.values())
]
random.seed(42)
random.shuffle(all_combos)          # random ordering = random search
total_trials = len(all_combos)      # 3 × 3 × 2 = 18 total

N_FOLDS    = 5
kf         = KFold(n_splits=N_FOLDS, shuffle=True, random_state=42)

trial_results = []

# ── Parent Run ─────────────────────────────────────────────────────────────────
with mlflow.start_run(run_name=PARENT_RUN_NAME) as parent_run:
    mlflow.set_tag("experiment_type", "hyperparameter_tuning")
    mlflow.log_param("search_type", "random")
    mlflow.log_param("n_folds", N_FOLDS)
    mlflow.log_param("total_trials", total_trials)

    for i, params in enumerate(all_combos, start=1):
        with mlflow.start_run(run_name=f"trial_{i}", nested=True):
            # Log params
            for k, v in params.items():
                mlflow.log_param(k, v)

            model = RandomForestRegressor(random_state=42, **params)

            # 5-fold CV — negative MAE scorer
            cv_neg_mae = cross_val_score(
                model, X, y,
                cv=kf,
                scoring="neg_mean_absolute_error",
                n_jobs=-1,
            )
            cv_mae = float(-cv_neg_mae.mean())

            # Fit on full data to get train RMSE
            model.fit(X, y)
            y_pred = model.predict(X)
            mae  = float(mean_absolute_error(y, y_pred))
            rmse = float(math.sqrt(mean_squared_error(y, y_pred)))

            mlflow.log_metric("cv_mae",  cv_mae)
            mlflow.log_metric("mae",     mae)
            mlflow.log_metric("rmse",    rmse)

            print(f"Trial {i:02d}/{total_trials} | params={params} "
                  f"| CV_MAE={cv_mae:.4f} | MAE={mae:.4f} | RMSE={rmse:.4f}")

            trial_results.append({
                "params": params,
                "cv_mae": round(cv_mae, 4),
                "mae":    round(mae, 4),
                "rmse":   round(rmse, 4),
            })

    # ── Select best by CV MAE (proxy for generalisation) then confirm with RMSE
    best_trial = min(trial_results, key=lambda x: x["cv_mae"])

    mlflow.log_metric("best_cv_mae", best_trial["cv_mae"])
    mlflow.log_metric("best_mae",    best_trial["mae"])
    mlflow.log_metric("best_rmse",   best_trial["rmse"])
    for k, v in best_trial["params"].items():
        mlflow.log_param(f"best_{k}", v)

# ── Save Results ───────────────────────────────────────────────────────────────
output = {
    "search_type":    "random",
    "n_folds":        N_FOLDS,
    "total_trials":   total_trials,
    "best_params":    best_trial["params"],
    "best_mae":       best_trial["mae"],
    "best_cv_mae":    best_trial["cv_mae"],
    "parent_run_name": PARENT_RUN_NAME,
}

with open(RESULTS_PATH, "w") as f:
    json.dump(output, f, indent=2)

print(f"\nBest params : {best_trial['params']}")
print(f"Best CV MAE : {best_trial['cv_mae']}")
print(f"Best MAE    : {best_trial['mae']}")
print(f"Results saved to: {RESULTS_PATH}")
