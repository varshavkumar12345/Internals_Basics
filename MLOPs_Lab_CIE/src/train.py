import os
import json
import math
import joblib
import pandas as pd
import numpy as np
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
import mlflow
import mlflow.sklearn

# ── Paths ──────────────────────────────────────────────────────────────────────
BASE_DIR     = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH    = os.path.join(BASE_DIR, "data", "training_data.csv")
RESULTS_DIR  = os.path.join(BASE_DIR, "results")
RESULTS_PATH = os.path.join(RESULTS_DIR, "step1_s1.json")
MODELS_DIR   = os.path.join(BASE_DIR, "models")
MLFLOW_DB    = os.path.join(BASE_DIR, "mlflow.db")

os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(MODELS_DIR,  exist_ok=True)

# ── MLflow Tracking URI (local, inside MLOPs_Lab_CIE/) ────────────────────────
mlflow.set_tracking_uri(f"sqlite:///{MLFLOW_DB}")

# ── Load Data ──────────────────────────────────────────────────────────────────
df = pd.read_csv(DATA_PATH)
TARGET   = "review_turnaround_hours"
FEATURES = [c for c in df.columns if c != TARGET]

X = df[FEATURES].values
y = df[TARGET].values

# ── MLflow Experiment ──────────────────────────────────────────────────────────
EXPERIMENT_NAME = "mergegate-review-turnaround-hours"
mlflow.set_experiment(EXPERIMENT_NAME)

# ── Model Definitions ──────────────────────────────────────────────────────────
models = {
    "Ridge": Ridge(alpha=1.0, fit_intercept=True, max_iter=1000),
    "RandomForest": RandomForestRegressor(
        n_estimators=100, max_depth=None, min_samples_split=2,
        min_samples_leaf=1, random_state=42
    ),
}

results = []

for model_name, model in models.items():
    with mlflow.start_run(run_name=model_name):
        mlflow.set_tag("experiment_type", "baseline_comparison")

        model.fit(X, y)
        y_pred = model.predict(X)

        mae  = float(mean_absolute_error(y, y_pred))
        rmse = float(math.sqrt(mean_squared_error(y, y_pred)))

        # Log all hyperparameters
        for k, v in model.get_params().items():
            mlflow.log_param(k, v)

        mlflow.log_metric("mae",  mae)
        mlflow.log_metric("rmse", rmse)

        # Log model artifact to MLflow
        mlflow.sklearn.log_model(model, name=model_name)

        # Save model locally to models/
        joblib.dump(model, os.path.join(MODELS_DIR, f"{model_name}.pkl"))

        print(f"[{model_name}]  MAE={mae:.4f}  RMSE={rmse:.4f}")
        results.append({"name": model_name, "mae": round(mae, 4), "rmse": round(rmse, 4)})

# ── Select Best Model by RMSE ──────────────────────────────────────────────────
best = min(results, key=lambda x: x["rmse"])

output = {
    "experiment_name":   EXPERIMENT_NAME,
    "models":            results,
    "best_model":        best["name"],
    "best_metric_name":  "rmse",
    "best_metric_value": best["rmse"],
}

with open(RESULTS_PATH, "w") as f:
    json.dump(output, f, indent=2)

print(f"\nBest model: {best['name']} (RMSE={best['rmse']})")
print(f"Results saved to: {RESULTS_PATH}")
