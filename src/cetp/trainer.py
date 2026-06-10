"""
BYOD trainer: fine-tunes a custom model on company-supplied data
using warm-start hyperparameter transfer from the base model.
"""

import os
from pathlib import Path
from typing import Optional

CETP_DIR = Path.home() / ".cetp"
MIN_R2_THRESHOLD = 0.80

BASE_HYPERPARAMS_PATH = (
    Path(__file__).parent.parent.parent / "model" / "artifacts" / "base_hyperparams.json"
)


class LowAccuracyWarning(UserWarning):
    """Emitted when the trained model's R² falls below MIN_R2_THRESHOLD."""

    pass


class CETTrainer:
    """
    Trains a custom CETP model on company-supplied CSV data.
    Uses warm-start hyperparameter transfer from the base model when available.
    """

    def __init__(self, base_hyperparams_path: Optional[str] = None) -> None:
        """
        Initialise the trainer.

        Args:
            base_hyperparams_path: Path to the base hyperparameters JSON file.
                                   Defaults to model/artifacts/base_hyperparams.json.
        """
        self._hyperparams_path = (
            Path(base_hyperparams_path) if base_hyperparams_path else BASE_HYPERPARAMS_PATH
        )
        self._hyperparams: dict = {}

    def train(
        self,
        csv_path: str,
        sla_path: Optional[str] = None,
        output_dir: Optional[str] = None,
    ) -> dict:
        """
        Train a custom model on the data at csv_path.

        Args:
            csv_path: Path to a validated CETP-schema CSV file.
            sla_path: Optional path to a company SLA JSON file.
            output_dir: Directory where the model artefact will be written.
                        Defaults to ~/.cetp/.

        Returns:
            dict with keys: model_path, r2_score, rmse, row_count, warnings.

        Raises:
            FileNotFoundError: If csv_path does not exist.
            FileNotFoundError: If base_hyperparams.json is missing (training
                               cannot proceed without base hyperparameters).
                               File needed: model/artifacts/base_hyperparams.json
            NotImplementedError: Full training pipeline pending base_hyperparams.json.
        """
        if not os.path.exists(csv_path):
            raise FileNotFoundError(f"Training CSV not found: {csv_path}")

        if not self._hyperparams_path.exists():
            raise FileNotFoundError(
                f"Base hyperparameters file not found: {self._hyperparams_path}. "
                "This file is required to warm-start BYOD training. "
                "It will be committed to model/artifacts/ after the initial training run."
            )

        # TODO: implement once base_hyperparams.json is available.
        #       Steps:
        #       1. Load hyperparams from self._hyperparams_path.
        #       2. Call validate_csv(csv_path) to ensure schema compliance.
        #       3. Call self._preprocess() to build feature/target arrays.
        #       4. Train XGBRegressor with warm-start hyperparams.
        #       5. Call self._evaluate() on a held-out validation split.
        #       6. If R² < MIN_R2_THRESHOLD, emit LowAccuracyWarning.
        #       7. Call self._save() to persist model and metadata.
        raise NotImplementedError(
            "train() requires model/artifacts/base_hyperparams.json. "
            "This file will be available after the initial base model training run."
        )

    def _preprocess(self, df) -> tuple:
        """
        Preprocess a DataFrame into feature matrix X and target vector y.

        Args:
            df: pandas DataFrame loaded from the validated CSV.

        Returns:
            (X, y) tuple of numpy arrays.

        Raises:
            NotImplementedError: Pending base_hyperparams.json availability.
        """
        # TODO: implement feature selection and one-hot encoding for disk_type.
        raise NotImplementedError(
            "_preprocess() is pending. Required file: model/artifacts/base_hyperparams.json"
        )

    def _evaluate(self, model, X_val, y_val) -> dict:
        """
        Evaluate a trained model on a validation split.

        Args:
            model: Trained XGBRegressor instance.
            X_val: Validation feature matrix.
            y_val: Validation target vector.

        Returns:
            dict with keys: r2_score, rmse, mae.

        Raises:
            NotImplementedError: Pending base_hyperparams.json availability.
        """
        # TODO: implement using sklearn.metrics r2_score, mean_squared_error.
        raise NotImplementedError(
            "_evaluate() is pending. Required file: model/artifacts/base_hyperparams.json"
        )

    def _save(self, model, metadata: dict, output_dir: Path) -> None:
        """
        Persist the trained model and its metadata to output_dir.

        Args:
            model: Trained model object to serialise with joblib.
            metadata: dict to write as a JSON sidecar file.
            output_dir: Directory path (will be created if absent).

        Raises:
            NotImplementedError: Pending base_hyperparams.json availability.
        """
        # TODO: implement with joblib.dump(model, output_dir / "model.pkl")
        #       and json.dump(metadata, output_dir / "model_metadata.json").
        raise NotImplementedError(
            "_save() is pending. Required file: model/artifacts/base_hyperparams.json"
        )
