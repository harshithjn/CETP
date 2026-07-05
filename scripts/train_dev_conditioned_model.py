"""
Trains and evaluates the dev-conditioned runtime predictor on
Dataset/dataset_dev_conditioned.csv (built by build_dev_conditioned_dataset.py).

Compares the same four algorithm families as the original base-model
training: Linear Regression, Decision Tree, Random Forest, XGBoost. Two
evaluation strategies are run and BOTH are reported in full, honestly:

  1. Random stratified split (stratified by target_tier) — the flattering,
     easy comparison. Train and test rows are drawn from the same
     distribution, so this mostly measures in-distribution fit.

  2. Leave-one-target-tier-out — train on rows whose target_tier is one of
     two tiers, test on rows whose target_tier is the third. This is the
     honest generalization check: can the model predict runtime on hardware
     it never saw as a *target* during training? Mirrors the original
     leave-one-tier-out finding referenced in predictor.py's
     check_confidence() docstring (R² of -15 to -33 when extrapolating).

Feature set (per the dev-conditioned schema — NOT the same as the shipped
base_model.pkl's current feature set):
    dev_cpu_count, dev_total_memory_mb, dev_runtime_sec,
    target_cpu_count, target_total_memory_mb,
    complexity_level, batch_size, num_iterations, model
Label: target_runtime_sec

dev_tier / target_tier are bookkeeping-only columns from the dataset build
step, used here purely to construct the stratified split and the
leave-one-target-tier-out folds. They are never passed to any pipeline as a
feature.
"""

import shutil
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATASET_PATH = PROJECT_ROOT / "Dataset" / "dataset_dev_conditioned.csv"
ARTIFACTS_DIR = PROJECT_ROOT / "model" / "artifacts"
MODEL_PATH = ARTIFACTS_DIR / "base_model.pkl"
TRAINING_RANGE_PATH = ARTIFACTS_DIR / "training_range.json"

FEATURE_COLUMNS = [
    "dev_cpu_count",
    "dev_total_memory_mb",
    "dev_runtime_sec",
    "target_cpu_count",
    "target_total_memory_mb",
    "complexity_level",
    "batch_size",
    "num_iterations",
    "model",
]
LABEL_COLUMN = "target_runtime_sec"
RANDOM_STATE = 42

# Same proven XGBoost config already documented in trainer.py's
# BASE_HYPERPARAMS. The other three models mirror its depth/estimator count
# where applicable so the comparison isn't skewed by wildly different
# capacity — there is no original multi-algorithm comparison script in this
# repo to replicate exactly, so these are disclosed, reasonable defaults,
# not a recovered ground truth.
MODEL_FACTORIES = {
    "Linear Regression": lambda: LinearRegression(),
    "Decision Tree": lambda: DecisionTreeRegressor(max_depth=6, random_state=RANDOM_STATE),
    "Random Forest": lambda: RandomForestRegressor(
        n_estimators=200, max_depth=6, random_state=RANDOM_STATE
    ),
    "XGBoost": lambda: XGBRegressor(
        n_estimators=200, max_depth=6, learning_rate=0.1, random_state=RANDOM_STATE
    ),
}


def make_pipeline(estimator) -> Pipeline:
    preprocessor = ColumnTransformer(
        transformers=[("cat", OneHotEncoder(handle_unknown="ignore"), ["model"])],
        remainder="passthrough",
    )
    return Pipeline(steps=[("prep", preprocessor), ("model", estimator)])


def evaluate(y_true, y_pred) -> dict:
    y_true_arr = np.asarray(y_true, dtype=float)
    y_pred_arr = np.asarray(y_pred, dtype=float)
    return {
        "mae": float(mean_absolute_error(y_true_arr, y_pred_arr)),
        "rmse": float(np.sqrt(mean_squared_error(y_true_arr, y_pred_arr))),
        "r2": float(r2_score(y_true_arr, y_pred_arr)),
        "mape": float(np.mean(np.abs((y_true_arr - y_pred_arr) / y_true_arr)) * 100),
    }


def fit_and_score(train_df: pd.DataFrame, test_df: pd.DataFrame, estimator_factory) -> dict:
    pipeline = make_pipeline(estimator_factory())
    pipeline.fit(train_df[FEATURE_COLUMNS], train_df[LABEL_COLUMN])
    y_pred = pipeline.predict(test_df[FEATURE_COLUMNS])
    return evaluate(test_df[LABEL_COLUMN], y_pred)


def print_table(headers: list[str], rows: list[list[str]]) -> None:
    widths = [
        max(len(str(headers[i])), *(len(str(row[i])) for row in rows)) if rows else len(headers[i])
        for i in range(len(headers))
    ]
    line = "  ".join(str(h).ljust(w) for h, w in zip(headers, widths))
    print(line)
    print("-" * len(line))
    for row in rows:
        print("  ".join(str(c).ljust(w) for c, w in zip(row, widths)))


def fmt_metrics_row(label_cols: list[str], metrics: dict) -> list[str]:
    return [
        *label_cols,
        f"{metrics['mae']:.4f}",
        f"{metrics['rmse']:.4f}",
        f"{metrics['r2']:.4f}",
        f"{metrics['mape']:.2f}",
    ]


def run_stratified_split(df: pd.DataFrame) -> dict:
    train_df, test_df = train_test_split(
        df, test_size=0.2, random_state=RANDOM_STATE, stratify=df["target_tier"]
    )
    results = {}
    rows = []
    for name, factory in MODEL_FACTORIES.items():
        metrics = fit_and_score(train_df, test_df, factory)
        results[name] = metrics
        rows.append(fmt_metrics_row([name], metrics))

    print()
    print("=" * 78)
    print("EVALUATION 1 — Random stratified split (80/20, stratified by target_tier)")
    print(f"  train rows: {len(train_df)}   test rows: {len(test_df)}")
    print("=" * 78)
    print_table(["Model", "MAE(s)", "RMSE(s)", "R2", "MAPE(%)"], rows)
    return results


def run_leave_one_target_tier_out(df: pd.DataFrame) -> dict:
    tiers = sorted(df["target_tier"].unique())
    results = {name: {} for name in MODEL_FACTORIES}
    rows = []
    for held_out in tiers:
        train_df = df[df["target_tier"] != held_out]
        test_df = df[df["target_tier"] == held_out]
        for name, factory in MODEL_FACTORIES.items():
            metrics = fit_and_score(train_df, test_df, factory)
            results[name][held_out] = metrics
            rows.append(fmt_metrics_row([name, held_out, len(train_df), len(test_df)], metrics))

    print()
    print("=" * 88)
    print("EVALUATION 2 — Leave-one-target-tier-out (honest generalization check)")
    print("=" * 88)
    print_table(
        ["Model", "Held-out target tier", "Train rows", "Test rows", "MAE(s)", "RMSE(s)", "R2", "MAPE(%)"],
        rows,
    )

    print()
    print("Per-model average across the 3 held-out tiers:")
    avg_rows = []
    for name, per_tier in results.items():
        avg_r2 = float(np.mean([m["r2"] for m in per_tier.values()]))
        avg_mae = float(np.mean([m["mae"] for m in per_tier.values()]))
        avg_rmse = float(np.mean([m["rmse"] for m in per_tier.values()]))
        avg_mape = float(np.mean([m["mape"] for m in per_tier.values()]))
        avg_rows.append([name, f"{avg_mae:.4f}", f"{avg_rmse:.4f}", f"{avg_r2:.4f}", f"{avg_mape:.2f}"])
    print_table(["Model", "avg MAE(s)", "avg RMSE(s)", "avg R2", "avg MAPE(%)"], avg_rows)

    return results


def backup_current_model() -> Path:
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    backup_path = ARTIFACTS_DIR / f"base_model.pre_dev_conditioned.{timestamp}.pkl.bak"
    if MODEL_PATH.exists():
        shutil.copy2(MODEL_PATH, backup_path)
        return backup_path
    return None


def backup_current_training_range() -> Path:
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    backup_path = ARTIFACTS_DIR / f"training_range.pre_dev_conditioned.{timestamp}.json.bak"
    if TRAINING_RANGE_PATH.exists():
        shutil.copy2(TRAINING_RANGE_PATH, backup_path)
        return backup_path
    return None


def main() -> None:
    if not DATASET_PATH.exists():
        raise FileNotFoundError(
            f"{DATASET_PATH} not found — run build_dev_conditioned_dataset.py first."
        )
    df = pd.read_csv(DATASET_PATH)

    print(f"Loaded {len(df)} paired rows from {DATASET_PATH}")

    stratified_results = run_stratified_split(df)
    loto_results = run_leave_one_target_tier_out(df)

    # Model selection is based on the LOTO average R2, not the stratified-split
    # number — the stratified split mixes tiers freely between train/test, so
    # it does not test whether the model generalizes to an unseen target tier,
    # which is the failure mode this whole exercise cares about.
    best_name = max(
        loto_results,
        key=lambda name: np.mean([m["r2"] for m in loto_results[name].values()]),
    )
    best_avg_r2 = float(np.mean([m["r2"] for m in loto_results[best_name].values()]))
    print()
    print(f"Selected '{best_name}' for the saved artifact — best average LOTO R2 ({best_avg_r2:.4f}).")
    print(
        "NOTE: a best-of-four LOTO R2 is not the same as 'generalizes well' in "
        "absolute terms — see the per-tier breakdown above before treating this "
        "number as reassuring."
    )

    model_backup = backup_current_model()
    range_backup = backup_current_training_range()
    print()
    if model_backup:
        print(f"Backed up current base_model.pkl -> {model_backup}")
    else:
        print("No existing base_model.pkl found to back up.")
    if range_backup:
        print(f"Backed up current training_range.json -> {range_backup}")
    else:
        print("No existing training_range.json found to back up.")

    final_pipeline = make_pipeline(MODEL_FACTORIES[best_name]())
    final_pipeline.fit(df[FEATURE_COLUMNS], df[LABEL_COLUMN])
    joblib.dump(final_pipeline, MODEL_PATH)
    print(f"Saved retrained '{best_name}' pipeline -> {MODEL_PATH}")

    training_range = {
        "dev_cpu_count": {"min": int(df["dev_cpu_count"].min()), "max": int(df["dev_cpu_count"].max())},
        "dev_total_memory_mb": {
            "min": float(df["dev_total_memory_mb"].min()),
            "max": float(df["dev_total_memory_mb"].max()),
        },
        "dev_runtime_sec": {
            "min": float(df["dev_runtime_sec"].min()),
            "max": float(df["dev_runtime_sec"].max()),
        },
        "target_cpu_count": {
            "min": int(df["target_cpu_count"].min()),
            "max": int(df["target_cpu_count"].max()),
        },
        "target_total_memory_mb": {
            "min": float(df["target_total_memory_mb"].min()),
            "max": float(df["target_total_memory_mb"].max()),
        },
        "batch_size": {"min": int(df["batch_size"].min()), "max": int(df["batch_size"].max())},
        "num_iterations": {
            "min": int(df["num_iterations"].min()),
            "max": int(df["num_iterations"].max()),
        },
        "known_models": sorted(df["model"].unique().tolist()),
        "known_tiers": sorted(set(df["dev_tier"].unique().tolist()) | set(df["target_tier"].unique().tolist())),
    }
    import json

    with open(TRAINING_RANGE_PATH, "w", encoding="utf-8") as fh:
        json.dump(training_range, fh, indent=2)
    print(f"Wrote updated training_range.json -> {TRAINING_RANGE_PATH}")

    print()
    print(
        "REMINDER: predictor.py, cli.py, and api/main.py were NOT modified in this "
        "pass. They still build the OLD feature row (cpu_count, total_memory_mb, "
        "complexity_level, batch_size, num_iterations, model) and read the OLD "
        "training_range.json keys (cpu_count, total_memory_mb). Both are now "
        "structurally incompatible with this new artifact and its training_range.json "
        "until a future pass updates them to collect a dev observation and build the "
        "new 9-feature row."
    )


if __name__ == "__main__":
    main()
