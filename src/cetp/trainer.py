"""
BYOD trainer: fits a custom model on company-supplied data using the same
proven hyperparameter configuration as the shipped base model.
"""

import json
import warnings
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from xgboost import XGBRegressor

from cetp.validator import ValidationError, validate_csv

CETP_DIR = Path.home() / ".cetp"
MIN_R2_THRESHOLD = 0.80

# Extracted directly from the trained base model (model/artifacts/base_model.pkl).
# These are the known-good hyperparameters — BYOD training reuses this exact
# configuration rather than re-tuning per company.
BASE_HYPERPARAMS = {
    "n_estimators": 200,
    "max_depth": 6,
    "learning_rate": 0.1,
    "random_state": 42,
}

_FEATURE_COLUMNS = [
    "cpu_count",
    "total_memory_mb",
    "complexity_level",
    "batch_size",
    "num_iterations",
    "model",
]


class LowAccuracyWarning(UserWarning):
    """Emitted when the trained model's R² falls below MIN_R2_THRESHOLD."""

    pass


class CETPTrainer:
    """
    Trains a custom CETP model on company-supplied CSV data, reusing the base
    model's proven hyperparameters rather than tuning from scratch.
    """

    def __init__(self, output_dir: Optional[str] = None) -> None:
        """
        Args:
            output_dir: Directory where trained artefacts are written.
                        Defaults to ~/.cetp/. Created if it doesn't exist.
        """
        self.output_dir = Path(output_dir) if output_dir is not None else CETP_DIR
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def train(self, csv_path: str, sla_path: Optional[str] = None) -> dict:
        """
        Validate, train, evaluate, and persist a custom model on csv_path.

        Raises:
            FileNotFoundError: If csv_path does not exist (raised by validate_csv).
            ValidationError: If the CSV violates the CETP schema. All violations
                             are printed before the exception is re-raised.
            InsufficientDataError: If csv_path has fewer than MIN_ROW_COUNT rows
                                   (a subclass of ValidationError; caught and
                                   printed the same way).
        """
        try:
            validation_result = validate_csv(csv_path)
        except ValidationError as exc:
            print(f"✗ Training data failed validation ({len(exc.violations)} issue(s) found):")
            for violation in exc.violations:
                print(f"  - {violation}")
            raise

        df = pd.read_csv(csv_path)

        X = df[_FEATURE_COLUMNS]
        y = df["runtime_sec"]

        X_train, X_val, y_train, y_val = train_test_split(
            X,
            y,
            test_size=0.2,
            random_state=BASE_HYPERPARAMS["random_state"],
            stratify=df["model"],
        )

        preprocessor = ColumnTransformer(
            transformers=[("cat", OneHotEncoder(handle_unknown="ignore"), ["model"])],
            remainder="passthrough",
        )

        # "Warm start" here means re-using the SAME PROVEN hyperparameter
        # configuration the base model was trained and validated with
        # (BASE_HYPERPARAMS) — it is NOT literal weight transfer or continued
        # boosting from the base model's trees. Each custom model is fit from
        # scratch on the company's own data.
        pipeline = Pipeline(
            steps=[
                ("prep", preprocessor),
                ("model", XGBRegressor(**BASE_HYPERPARAMS)),
            ]
        )
        pipeline.fit(X_train, y_train)

        y_pred = pipeline.predict(X_val)
        metrics = self._evaluate(y_val, y_pred)

        low_accuracy = metrics["r2"] < MIN_R2_THRESHOLD
        if low_accuracy:
            message = (
                f"validation R^2={metrics['r2']:.3f} is below the {MIN_R2_THRESHOLD} "
                "accuracy threshold. Saving the model anyway — a lower-accuracy "
                "custom model may still be preferable to the generic base model."
            )
            warnings.warn(f"LowAccuracyWarning: {message}", LowAccuracyWarning, stacklevel=2)
            print(f"⚠ LowAccuracyWarning: {message}")

        model_path = self.output_dir / "custom_model.pkl"
        joblib.dump(pipeline, model_path)

        training_range = {
            "cpu_count": {"min": int(df["cpu_count"].min()), "max": int(df["cpu_count"].max())},
            "total_memory_mb": {
                "min": float(df["total_memory_mb"].min()),
                "max": float(df["total_memory_mb"].max()),
            },
            "batch_size": {"min": int(df["batch_size"].min()), "max": int(df["batch_size"].max())},
            "num_iterations": {
                "min": int(df["num_iterations"].min()),
                "max": int(df["num_iterations"].max()),
            },
            "known_models": sorted(df["model"].unique().tolist()),
            # BYOD CSVs carry no hardware-tier column (see REQUIRED_COLUMNS in
            # validator.py), so this custom model has no tier information to report.
            "known_tiers": [],
        }
        training_range_path = self.output_dir / "custom_training_range.json"
        with open(training_range_path, "w", encoding="utf-8") as fh:
            json.dump(training_range, fh, indent=2)

        metadata = {
            "trained_at": datetime.now(timezone.utc).isoformat(),
            "source_csv": str(csv_path),
            "row_count": int(validation_result["row_count"]),
            "metrics": metrics,
            "hyperparameters": BASE_HYPERPARAMS,
        }
        metadata_path = self.output_dir / "custom_model_meta.json"
        with open(metadata_path, "w", encoding="utf-8") as fh:
            json.dump(metadata, fh, indent=2)

        return {
            "model_path": str(model_path),
            "training_range_path": str(training_range_path),
            "metadata_path": str(metadata_path),
            "row_count": int(validation_result["row_count"]),
            "metrics": metrics,
            "hyperparameters": BASE_HYPERPARAMS,
            "low_accuracy_warning": low_accuracy,
        }

    @staticmethod
    def _evaluate(y_true, y_pred) -> dict:
        """Compute MAE, RMSE, R², and MAPE (%) for a validation split."""
        y_true_arr = np.asarray(y_true, dtype=float)
        y_pred_arr = np.asarray(y_pred, dtype=float)

        mae = float(mean_absolute_error(y_true_arr, y_pred_arr))
        rmse = float(np.sqrt(mean_squared_error(y_true_arr, y_pred_arr)))
        r2 = float(r2_score(y_true_arr, y_pred_arr))
        mape = float(np.mean(np.abs((y_true_arr - y_pred_arr) / y_true_arr)) * 100)

        return {"mae": mae, "rmse": rmse, "r2": r2, "mape": mape}
