"""
SHAP explainer: computes feature attributions for individual predictions
using TreeExplainer for gradient boosted tree models.
"""

import numpy as np


class CETExplainer:
    """
    Wraps shap.TreeExplainer to produce per-feature attribution scores
    for individual CETP predictions and global feature importance summaries.
    """

    def __init__(self, model, feature_names: list) -> None:
        """
        Initialise the explainer with a trained tree model.

        Args:
            model: Trained XGBRegressor (or any tree model supported by shap).
            feature_names: Ordered list of feature names matching the model's
                           input column order.

        Raises:
            NotImplementedError: Until the model artefact is available.
                                 File needed: model/artifacts/base_model.pkl
        """
        # TODO: implement once base_model.pkl is available.
        #       import shap
        #       self._explainer = shap.TreeExplainer(model)
        #       self._feature_names = feature_names
        raise NotImplementedError(
            "__init__() requires a trained model passed in. "
            "File needed: model/artifacts/base_model.pkl"
        )

    def explain(self, feature_vector: np.ndarray, top_n: int = 3) -> list:
        """
        Compute SHAP values for a single prediction and return the top_n
        most impactful features.

        Args:
            feature_vector: 1-D numpy array of input features in model order.
            top_n: Number of top attributions to return (default 3).

        Returns:
            List of dicts sorted by abs(shap_value) descending, each with:
                feature (str): feature name
                value (float): raw feature value
                shap_value (float): SHAP attribution

        Raises:
            NotImplementedError: Until the model artefact is available.
        """
        # TODO: implement once __init__() is functional.
        #       values = self._explainer.shap_values(feature_vector.reshape(1, -1))[0]
        #       pairs = sorted(zip(self._feature_names, feature_vector, values),
        #                      key=lambda t: abs(t[2]), reverse=True)
        #       return [{"feature": f, "value": float(v), "shap_value": float(s)}
        #               for f, v, s in pairs[:top_n]]
        raise NotImplementedError(
            "explain() requires a trained model artefact. "
            "File needed: model/artifacts/base_model.pkl"
        )

    def explain_global(self, X: np.ndarray) -> dict:
        """
        Compute global mean absolute SHAP values across a dataset.

        Args:
            X: 2-D numpy array of shape (n_samples, n_features).

        Returns:
            dict mapping each feature name to its mean absolute SHAP value,
            sorted by importance descending.

        Raises:
            NotImplementedError: Until the model artefact is available.
        """
        # TODO: implement once __init__() is functional.
        #       shap_matrix = self._explainer.shap_values(X)
        #       mean_abs = np.abs(shap_matrix).mean(axis=0)
        #       return dict(sorted(zip(self._feature_names, mean_abs.tolist()),
        #                          key=lambda t: t[1], reverse=True))
        raise NotImplementedError(
            "explain_global() requires a trained model artefact. "
            "File needed: model/artifacts/base_model.pkl"
        )
