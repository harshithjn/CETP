"""
Prediction engine: loads the trained model artefact and produces
runtime estimates with confidence intervals, SLA flags, and
SHAP-based explanations.
"""

import json
import logging
from pathlib import Path
from typing import Optional

import joblib
import pandas as pd

from cetp.workload_config import VALID_MODELS, get_workload_params

logger = logging.getLogger(__name__)

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

CETP_DIR = Path.home() / ".cetp"
CUSTOM_MODEL_PATH = CETP_DIR / "custom_model.pkl"
CUSTOM_TRAINING_RANGE_PATH = CETP_DIR / "custom_training_range.json"

BASE_MODEL_PATH = _PROJECT_ROOT / "model" / "artifacts" / "base_model.pkl"
BASE_TRAINING_RANGE_PATH = _PROJECT_ROOT / "model" / "artifacts" / "training_range.json"

DEFAULT_SLA_CONFIG_PATH = _PROJECT_ROOT / "sla_defaults.json"

SLA_FLAGS = ("GREEN", "YELLOW", "RED")

# Derived from held-out test residuals during model evaluation. A true quantile
# regression (or conformal prediction) would give a more rigorous interval, but
# this fixed residual-std approximation is fast and defensible for now.
RESIDUAL_STD = 15.2
_Z_90 = 1.645


class CETPPredictor:
    """
    Loads the trained CETP model artefact and predicts production runtime
    with confidence intervals, SLA traffic-light flags, and SHAP attributions.
    """

    def __init__(
        self,
        model_path: Optional[str] = None,
        training_range_path: Optional[str] = None,
        sla_config_path: Optional[str] = None,
    ) -> None:
        """
        Resolve and load all artefacts needed for prediction.

        model_path / training_range_path priority:
            1. explicit argument, if given (must exist or FileNotFoundError is raised)
            2. ~/.cetp/custom_{model,training_range}.{pkl,json} (BYOD custom artefact)
            3. model/artifacts/{base_model.pkl,training_range.json} (package base artefact)

        Raises:
            FileNotFoundError: if no model or training range file can be found
                               at any priority level.
        """
        self._model_path = self._resolve_path(
            explicit_path=model_path,
            custom_path=CUSTOM_MODEL_PATH,
            base_path=BASE_MODEL_PATH,
            what="model artefact",
        )
        self._training_range_path = self._resolve_path(
            explicit_path=training_range_path,
            custom_path=CUSTOM_TRAINING_RANGE_PATH,
            base_path=BASE_TRAINING_RANGE_PATH,
            what="training_range.json",
        )

        with open(self._training_range_path, encoding="utf-8") as fh:
            self.training_range: dict = json.load(fh)

        self._sla_config_path = (
            Path(sla_config_path) if sla_config_path is not None else DEFAULT_SLA_CONFIG_PATH
        )
        self._sla_config: dict = {}
        if self._sla_config_path.exists():
            with open(self._sla_config_path, encoding="utf-8") as fh:
                self._sla_config = json.load(fh)

        self.pipeline = None
        self.load_model()

    @staticmethod
    def _resolve_path(
        explicit_path: Optional[str], custom_path: Path, base_path: Path, what: str
    ) -> Path:
        if explicit_path is not None:
            path = Path(explicit_path)
            if not path.exists():
                raise FileNotFoundError(f"Specified {what} path does not exist: {path}")
            return path

        if custom_path.exists():
            return custom_path

        if base_path.exists():
            return base_path

        raise FileNotFoundError(
            f"No {what} found. Checked:\n"
            f"  - custom: {custom_path}\n"
            f"  - base:   {base_path}\n"
            "Run `cetp train` to produce a custom artefact, or ensure the "
            "package's base artefact is present at the path above."
        )

    def load_model(self) -> None:
        """Load the sklearn Pipeline from disk via joblib."""
        self.pipeline = joblib.load(self._model_path)

    def check_confidence(self, cpu_count: int, total_memory_mb: float, model_name: str) -> dict:
        """
        Check whether the requested inputs fall inside the training data's
        hardware range and known model set. Predictions requested outside this
        range showed R^2 of -15 to -33 in leave-one-tier-out validation, so an
        EXTRAPOLATED status here must always be surfaced to the caller.
        """
        warnings: list = []

        cpu_range = self.training_range.get("cpu_count", {})
        cpu_min, cpu_max = cpu_range.get("min"), cpu_range.get("max")
        if cpu_min is not None and cpu_max is not None and not (cpu_min <= cpu_count <= cpu_max):
            warnings.append(
                f"cpu_count={cpu_count} is outside the trained range [{cpu_min}, {cpu_max}]"
            )

        mem_range = self.training_range.get("total_memory_mb", {})
        mem_min, mem_max = mem_range.get("min"), mem_range.get("max")
        if (
            mem_min is not None
            and mem_max is not None
            and not (mem_min <= total_memory_mb <= mem_max)
        ):
            warnings.append(
                f"total_memory_mb={total_memory_mb} is outside the trained range "
                f"[{mem_min}, {mem_max}]"
            )

        known_models = self.training_range.get("known_models", [])
        if model_name not in known_models:
            warnings.append(
                f"model '{model_name}' was not among the models seen during training "
                f"({known_models})"
            )

        status = "EXTRAPOLATED" if warnings else "RELIABLE"
        return {"status": status, "warnings": warnings}

    def predict(
        self,
        model_name: str,
        complexity_level: int,
        cpu_count: int,
        total_memory_mb: float,
    ) -> dict:
        """
        Predict production runtime for the given model, complexity level, and
        hardware profile. Raises ValueError for invalid model_name or
        complexity_level; wraps any other failure in a RuntimeError with context.
        """
        if model_name not in VALID_MODELS:
            raise ValueError(f"Invalid model '{model_name}', must be one of {VALID_MODELS}")

        if (
            not isinstance(complexity_level, int)
            or isinstance(complexity_level, bool)
            or not (1 <= complexity_level <= 5)
        ):
            raise ValueError(
                f"Invalid complexity_level {complexity_level!r}, must be an integer in [1, 5]"
            )

        try:
            batch_size, num_iterations = get_workload_params(model_name, complexity_level)

            confidence = self.check_confidence(cpu_count, total_memory_mb, model_name)

            row = pd.DataFrame(
                [
                    {
                        "cpu_count": cpu_count,
                        "total_memory_mb": total_memory_mb,
                        "complexity_level": complexity_level,
                        "batch_size": batch_size,
                        "num_iterations": num_iterations,
                        "model": model_name,
                    }
                ]
            )

            point_estimate = float(self.pipeline.predict(row)[0])
            lower = max(0.0, point_estimate - _Z_90 * RESIDUAL_STD)
            upper = point_estimate + _Z_90 * RESIDUAL_STD

            return {
                "predicted_runtime_sec": point_estimate,
                "confidence_interval": [lower, upper],
                "confidence_status": confidence["status"],
                "confidence_warnings": confidence["warnings"],
                "model_used": model_name,
                "complexity_level": complexity_level,
                "batch_size": batch_size,
                "num_iterations": num_iterations,
            }
        except (ValueError, KeyError):
            raise
        except Exception as exc:
            raise RuntimeError(f"Prediction failed for model={model_name!r}: {exc}") from exc

    def compute_sla_flag(self, predicted_upper: float, model_name: str) -> str:
        """
        Map the upper bound of the prediction interval to a GREEN/YELLOW/RED
        SLA flag using the model-specific thresholds loaded from sla_config.
        """
        if model_name not in self._sla_config:
            raise KeyError(
                f"No SLA configuration found for model '{model_name}'. "
                f"Configured models: {list(self._sla_config.keys())}"
            )

        thresholds = self._sla_config[model_name]
        warn_at = thresholds["warn_at_sec"]
        sla_limit = thresholds["sla_runtime_sec"]

        if predicted_upper <= warn_at:
            return "GREEN"
        if predicted_upper <= sla_limit:
            return "YELLOW"
        return "RED"

    def get_shap_explanation(
        self,
        model_name: str,
        complexity_level: int,
        cpu_count: int,
        total_memory_mb: float,
        top_n: int = 3,
    ) -> dict:
        """
        Return the top_n SHAP feature attributions for a single prediction,
        plus the explainer's base_value (its expected output over the
        training set) — base_value + sum(all feature shap_values) must
        reconstruct the model's raw prediction.
        Never raises: any failure (missing shap, transform error, etc.) is
        logged and results in {"base_value": 0.0, "features": []} so a
        broken explainer never takes down a prediction request.
        """
        try:
            import shap

            batch_size, num_iterations = get_workload_params(model_name, complexity_level)
            row = pd.DataFrame(
                [
                    {
                        "cpu_count": cpu_count,
                        "total_memory_mb": total_memory_mb,
                        "complexity_level": complexity_level,
                        "batch_size": batch_size,
                        "num_iterations": num_iterations,
                        "model": model_name,
                    }
                ]
            )

            preprocessor = self.pipeline.named_steps["prep"]
            xgb_model = self.pipeline.named_steps["model"]

            transformed = preprocessor.transform(row)
            feature_names = preprocessor.get_feature_names_out()

            explainer = shap.TreeExplainer(xgb_model)
            # expected_value is a plain float for some SHAP/model versions and a
            # length-1 ndarray for others (single-output regression) — numpy 2.x
            # raises on float() of a non-0-d array, so unwrap explicitly.
            raw_base_value = explainer.expected_value
            base_value = (
                float(raw_base_value[0])
                if hasattr(raw_base_value, "__len__")
                else float(raw_base_value)
            )
            shap_values = explainer.shap_values(transformed)[0]

            # One-hot encoding splits 'model' into one binary column per category
            # (cat__model_resnet18, cat__model_resnet50, ...), and SHAP assigns
            # every one of those columns its own value — including the ones that
            # are OFF (0) for this row. Naively ranking all raw columns can then
            # surface an OFF category (e.g. cat__model_distilbert while predicting
            # resnet18) as a top contributor, which is incoherent to show a user.
            # SHAP values are additive, so the one-hot group is summed back into a
            # single "model" contribution, labelled with the model actually being
            # predicted — never a different category.
            model_prefix = "cat__model_"
            model_shap_total = 0.0
            other_features = []
            for name, value in zip(feature_names, shap_values):
                if name.startswith(model_prefix):
                    model_shap_total += float(value)
                else:
                    other_features.append((name, float(value)))
            other_features.append((f"{model_prefix}{model_name}", model_shap_total))

            ranked = sorted(other_features, key=lambda pair: abs(pair[1]), reverse=True)[:top_n]

            return {
                "base_value": base_value,
                "features": [
                    {
                        "feature": str(name),
                        "shap_value": float(value),
                        "direction": "increases_runtime" if value > 0 else "decreases_runtime",
                    }
                    for name, value in ranked
                ],
            }
        except Exception as exc:
            logger.warning("SHAP explanation failed: %s", exc)
            return {"base_value": 0.0, "features": []}
