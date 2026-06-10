"""
Prediction engine: loads the trained model artefact and produces
runtime estimates with confidence intervals, SLA flags, and
SHAP-based explanations.
"""
import json
import os
from pathlib import Path
from typing import Optional

CETP_DIR = Path.home() / ".cetp"
BASE_MODEL_PATH = Path(__file__).parent.parent.parent / "model" / "artifacts" / "base_model.pkl"

SLA_FLAGS = ("GREEN", "YELLOW", "RED")


class CETPPredictor:
    """
    Loads the trained CETP model artefact and predicts production runtime
    with confidence intervals, SLA traffic-light flags, and SHAP attributions.
    """

    def __init__(self, sla_config_path: Optional[str] = None) -> None:
        """
        Initialise the predictor.

        Args:
            sla_config_path: Path to a JSON file with SLA thresholds.
                             Falls back to sla_defaults.json shipped with the package.
        """
        self._model = None
        self._feature_names: list[str] = []
        self._sla_config: dict = {}
        self._model_metadata: dict = {}

        # Resolve SLA config
        if sla_config_path is None:
            default_path = Path(__file__).parent.parent.parent / "sla_defaults.json"
            sla_config_path = str(default_path)
        self._sla_config_path = sla_config_path

        if os.path.exists(sla_config_path):
            with open(sla_config_path) as fh:
                self._sla_config = json.load(fh)

    def load_model(self) -> None:
        """
        Load the trained model from disk.

        Searches for a custom model in ~/.cetp/model.pkl first, then falls
        back to the base model shipped with the package.

        Raises:
            FileNotFoundError: If neither the custom nor the base model artefact
                               exists. The base model is created during project
                               training (see trainer.py). File needed:
                               model/artifacts/base_model.pkl
        """
        # TODO: implement once model/artifacts/base_model.pkl is available.
        #       Steps:
        #       1. Check CETP_DIR / "model.pkl" for a custom BYOD model.
        #       2. Fall back to BASE_MODEL_PATH for the shipped base model.
        #       3. Load with joblib.load().
        #       4. Load accompanying metadata JSON (same stem, .json extension).
        raise NotImplementedError(
            "load_model() requires a trained model artefact. "
            "Run `cetp train` to create one, or place base_model.pkl in "
            f"{BASE_MODEL_PATH}"
        )

    def predict(self, feature_dict: dict) -> dict:
        """
        Predict production runtime for the given feature vector.

        Args:
            feature_dict: Dict containing all required CETP features.

        Returns:
            dict with keys:
                predicted_sec (float): point estimate
                lower_sec (float): lower bound of 90 % CI
                upper_sec (float): upper bound of 90 % CI
                sla_flag (str): GREEN | YELLOW | RED
                workload_type (str): echoed from input

        Raises:
            NotImplementedError: Until the model artefact is available and
                                 load_model() has been implemented.
        """
        # TODO: implement once load_model() is functional.
        #       Steps:
        #       1. Call load_model() if self._model is None.
        #       2. Build feature array in the correct column order.
        #       3. Call model.predict() for the point estimate.
        #       4. Derive CI using model's quantile regressors or +/- residual std.
        #       5. Call compute_sla_flag() with predicted_upper.
        raise NotImplementedError(
            "predict() requires a trained model artefact. "
            "File needed: model/artifacts/base_model.pkl"
        )

    def compute_sla_flag(self, predicted_upper: float, workload_type: str) -> str:
        """
        Compute the SLA traffic-light flag.

        Args:
            predicted_upper: Upper bound of the predicted runtime CI in seconds.
            workload_type: One of ML, DB, WEB.

        Returns:
            'GREEN'  — safely within SLA
            'YELLOW' — within SLA but past the warn threshold
            'RED'    — predicted to breach SLA

        Raises:
            NotImplementedError: Until the SLA config and model are integrated.
        """
        # TODO: implement once SLA config loading is complete.
        #       Logic:
        #       thresholds = self._sla_config.get(workload_type, {})
        #       if predicted_upper >= thresholds["sla_runtime_sec"]: return "RED"
        #       if predicted_upper >= thresholds["warn_at_sec"]: return "YELLOW"
        #       return "GREEN"
        raise NotImplementedError(
            "compute_sla_flag() requires self._sla_config to be populated. "
            "Load the SLA JSON file before calling this method."
        )

    def get_shap_explanation(self, feature_dict: dict) -> list:
        """
        Return the top-3 SHAP feature attributions for a single prediction.

        Args:
            feature_dict: Same feature dict passed to predict().

        Returns:
            List of dicts, each with keys: feature, value, shap_value.
            Sorted by abs(shap_value) descending. Top 3 only.

        Raises:
            NotImplementedError: Until the model artefact and explainer are ready.
                                 File needed: model/artifacts/base_model.pkl
        """
        # TODO: implement once load_model() is functional.
        #       Steps:
        #       1. Build feature array.
        #       2. Instantiate CETExplainer(self._model, self._feature_names).
        #       3. Call explainer.explain(feature_array, top_n=3).
        raise NotImplementedError(
            "get_shap_explanation() requires a trained model artefact. "
            "File needed: model/artifacts/base_model.pkl"
        )

    def get_active_model_info(self) -> dict:
        """
        Return metadata about the currently active model.

        Returns:
            dict with keys:
                model_path (str): absolute path to the loaded model file
                model_version (str): semantic version from metadata
                trained_on (str): ISO date string from metadata
                feature_count (int): number of input features
                custom_model (bool): True if a BYOD model is active
        """
        # TODO: implement once load_model() is functional.
        #       Return self._model_metadata enriched with path and custom_model flag.
        custom_model_path = CETP_DIR / "model.pkl"
        return {
            "model_path": str(custom_model_path) if custom_model_path.exists() else str(BASE_MODEL_PATH),
            "model_version": self._model_metadata.get("version", "unknown"),
            "trained_on": self._model_metadata.get("trained_on", "unknown"),
            "feature_count": self._model_metadata.get("feature_count", 0),
            "custom_model": custom_model_path.exists(),
        }
