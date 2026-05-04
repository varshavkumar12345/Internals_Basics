import os
import json
import math
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error
import mlflow
import mlflow.sklearn
from mlflow.tracking import MlflowClient

# ── Paths ──────────────────────────────────────────────────────────────────────
BASE_DIR        = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TRAIN_PATH      = os.path.join(BASE_DIR, "data", "training_data.csv")
NEW_DATA_PATH   = os.path.join(BASE_DIR, "data", "new_data.csv")
RESULTS_DIR     = os.path.join(BASE_DIR, "results")
RESULTS_PATH    = os.path.join(RESULTS_DIR, "step4_s8.json")
STEP1_JSON      = os.path.join(RESULTS_DIR, "step1_s1.json")
STEP2_JSON      = os.path.join(RESULTS_DIR, "step2_s2.json")

os.makedirs(RESULTS_DIR, exist_ok=True)

# ── Load best params from Task 2 (tune.py) ────────────────────────────────────
with open(STEP2_JSON) as f:
    tune_results = json.load(f)

best_params = tune_results["best_params"]
print(f"Using best params from Task 2: {best_params}")

# ── Load & Combine Data ────────────────────────────────────────────────────────
df_original = pd.read_csv(TRAIN_PATH)
df_new      = pd.read_csv(NEW_DATA_PATH)

original_rows = len(df_original)
new_rows      = len(df_new)

df_combined   = pd.concat([df_original, df_new], ignore_index=True)
combined_rows = len(df_combined)

print(f"Original rows : {original_rows}")
print(f"New rows      : {new_rows}")
print(f"Combined rows : {combined_rows}")

TARGET   = "review_turnaround_hours"
FEATURES = [c for c in df_combined.columns if c != TARGET]

X = df_combined[FEATURES].values
y = df_combined[TARGET].values

# ── Shared test split (80/20, same seed for fair comparison) ──────────────────
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ── Config ─────────────────────────────────────────────────────────────────────
EXPERIMENT_NAME  = "mergegate-review-turnaround-hours"
REGISTERED_NAME  = "mergegate-review-turnaround-hours-predictor"
MIN_IMPROVEMENT  = 1.0

mlflow.set_experiment(EXPERIMENT_NAME)
client = MlflowClient()

# ── Evaluate Champion on test set ─────────────────────────────────────────────
# Load champion model from registry (version 1)
champion_uri   = f"models:/{REGISTERED_NAME}/1"
champion_model = mlflow.sklearn.load_model(champion_uri)

y_pred_champion  = champion_model.predict(X_test)
champion_rmse    = float(math.sqrt(mean_squared_error(y_test, y_pred_champion)))
champion_mae     = float(mean_absolute_error(y_test, y_pred_champion))
print(f"\nChampion RMSE : {champion_rmse:.4f}  MAE : {champion_mae:.4f}")

# ── Retrain on Combined Data ───────────────────────────────────────────────────
with mlflow.start_run(run_name="retrain-combined") as retrain_run:
    mlflow.set_tag("experiment_type", "retraining")
    mlflow.set_tag("champion_version", "1")

    # Log data stats
    mlflow.log_param("original_data_rows", original_rows)
    mlflow.log_param("new_data_rows",      new_rows)
    mlflow.log_param("combined_data_rows", combined_rows)

    # Log best params
    for k, v in best_params.items():
        mlflow.log_param(k, v)

    retrained_model = RandomForestRegressor(random_state=42, **best_params)
    retrained_model.fit(X_train, y_train)

    y_pred_retrained = retrained_model.predict(X_test)
    retrained_rmse   = float(math.sqrt(mean_squared_error(y_test, y_pred_retrained)))
    retrained_mae    = float(mean_absolute_error(y_test, y_pred_retrained))

    mlflow.log_metric("champion_rmse",  champion_rmse)
    mlflow.log_metric("retrained_rmse", retrained_rmse)
    mlflow.log_metric("retrained_mae",  retrained_mae)

    improvement = round(champion_rmse - retrained_rmse, 4)
    mlflow.log_metric("rmse_improvement", improvement)

    print(f"Retrained RMSE : {retrained_rmse:.4f}  MAE : {retrained_mae:.4f}")
    print(f"Improvement    : {improvement:.4f}  (threshold = {MIN_IMPROVEMENT})")

    retrain_run_id = retrain_run.info.run_id

    # ── Promotion Decision ─────────────────────────────────────────────────────
    if improvement >= MIN_IMPROVEMENT:
        action = "promoted"
        mlflow.set_tag("promotion_decision", "promoted")

        # Log and register the retrained model
        mlflow.sklearn.log_model(retrained_model, name="RandomForest")
        new_version = mlflow.register_model(
            model_uri=f"runs:/{retrain_run_id}/RandomForest",
            name=REGISTERED_NAME,
        )
        print(f"Model PROMOTED as version {new_version.version}")
    else:
        action = "kept_champion"
        mlflow.set_tag("promotion_decision", "kept_champion")
        print("Champion RETAINED — improvement below threshold.")

# ── Save Results ───────────────────────────────────────────────────────────────
output = {
    "original_data_rows":      original_rows,
    "new_data_rows":           new_rows,
    "combined_data_rows":      combined_rows,
    "champion_rmse":           round(champion_rmse,   4),
    "retrained_rmse":          round(retrained_rmse,  4),
    "improvement":             improvement,
    "min_improvement_threshold": MIN_IMPROVEMENT,
    "action":                  action,
    "comparison_metric":       "rmse",
}

with open(RESULTS_PATH, "w") as f:
    json.dump(output, f, indent=2)

print(f"\nAction        : {action}")
print(f"Results saved to: {RESULTS_PATH}")
print(json.dumps(output, indent=2))
