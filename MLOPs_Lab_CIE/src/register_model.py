import os
import json
import mlflow
from mlflow.tracking import MlflowClient

# ── Paths ──────────────────────────────────────────────────────────────────────
BASE_DIR     = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS_DIR  = os.path.join(BASE_DIR, "results")
RESULTS_PATH = os.path.join(RESULTS_DIR, "step3_s6.json")

os.makedirs(RESULTS_DIR, exist_ok=True)

# ── Config ─────────────────────────────────────────────────────────────────────
EXPERIMENT_NAME    = "mergegate-review-turnaround-hours"
REGISTERED_NAME    = "mergegate-review-turnaround-hours-predictor"
SOURCE_METRIC      = "rmse"
MODEL_ARTIFACT_DIR = "RandomForest"   # artifact_path used in train.py

# ── MLflow Client ──────────────────────────────────────────────────────────────
mlflow.set_experiment(EXPERIMENT_NAME)
client = MlflowClient()

experiment = client.get_experiment_by_name(EXPERIMENT_NAME)
if experiment is None:
    raise RuntimeError(f"Experiment '{EXPERIMENT_NAME}' not found. Run train.py first.")

# ── Find Best Run by RMSE (lowest, non-nested runs only) ──────────────────────
runs = client.search_runs(
    experiment_ids=[experiment.experiment_id],
    filter_string=f"metrics.{SOURCE_METRIC} > 0",   # only runs that have rmse logged
    order_by=[f"metrics.{SOURCE_METRIC} ASC"],
    max_results=50,
)

# Filter out nested runs (they have a 'mlflow.parentRunId' tag)
top_level_runs = [
    r for r in runs
    if "mlflow.parentRunId" not in r.data.tags
]

if not top_level_runs:
    raise RuntimeError("No eligible top-level runs found with 'rmse' metric.")

best_run = top_level_runs[0]
best_run_id   = best_run.info.run_id
best_rmse     = best_run.data.metrics[SOURCE_METRIC]
best_run_name = best_run.data.tags.get("mlflow.runName", "unknown")

print(f"Best run  : {best_run_name}  (run_id={best_run_id})")
print(f"Best RMSE : {best_rmse:.4f}")

# ── Build Model URI and Register ───────────────────────────────────────────────
model_uri = f"runs:/{best_run_id}/{MODEL_ARTIFACT_DIR}"
print(f"Registering model from URI: {model_uri}")

model_version = mlflow.register_model(
    model_uri=model_uri,
    name=REGISTERED_NAME,
)

version_number = int(model_version.version)
print(f"Registered as version : {version_number}")

# ── Tag the version with the source run ───────────────────────────────────────
client.set_model_version_tag(
    name=REGISTERED_NAME,
    version=str(version_number),
    key="source_run_id",
    value=best_run_id,
)
client.set_model_version_tag(
    name=REGISTERED_NAME,
    version=str(version_number),
    key="source_metric",
    value=SOURCE_METRIC,
)
client.set_model_version_tag(
    name=REGISTERED_NAME,
    version=str(version_number),
    key="source_metric_value",
    value=str(round(best_rmse, 4)),
)

# ── Save Results ───────────────────────────────────────────────────────────────
output = {
    "registered_model_name": REGISTERED_NAME,
    "version":               version_number,
    "run_id":                best_run_id,
    "source_metric":         SOURCE_METRIC,
    "source_metric_value":   round(best_rmse, 4),
}

with open(RESULTS_PATH, "w") as f:
    json.dump(output, f, indent=2)

print(f"\nResults saved to: {RESULTS_PATH}")
print(json.dumps(output, indent=2))
