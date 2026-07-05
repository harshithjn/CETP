"""Comprehensive tests for the CETP FastAPI REST backend."""

import pytest
from fastapi.testclient import TestClient

from api.main import app

client = TestClient(app)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def valid_predict_payload():
    return {
        "model_name": "resnet18",
        "complexity_level": 3,
        "cpu_count": 4,
        "total_memory_mb": 8000.0,
    }


# ---------------------------------------------------------------------------
# Health endpoint
# ---------------------------------------------------------------------------


class TestHealth:
    def test_health_returns_200(self):
        response = client.get("/health")
        assert response.status_code == 200

    def test_health_returns_ok_status(self):
        response = client.get("/health")
        assert response.json()["status"] == "ok"

    def test_health_has_version(self):
        response = client.get("/health")
        assert response.json()["version"] == "0.1.0"

    def test_health_has_model_status(self):
        response = client.get("/health")
        assert "model_status" in response.json()

    def test_health_has_sla_config_loaded(self):
        response = client.get("/health")
        assert "sla_config_loaded" in response.json()

    def test_health_model_status_is_valid_value(self):
        response = client.get("/health")
        valid = {"base_model_loaded", "custom_model_loaded", "no_model"}
        assert response.json()["model_status"] in valid

    def test_health_never_returns_500(self):
        for _ in range(3):
            response = client.get("/health")
            assert response.status_code == 200

    def test_health_has_training_range_key(self):
        response = client.get("/health")
        assert "training_range" in response.json()

    def test_health_training_range_reports_known_models(self):
        """With the shipped base_model.pkl present, training_range should
        report the known ML models it was trained on."""
        response = client.get("/health")
        body = response.json()
        if body["model_status"] != "no_model":
            assert body["training_range"] is not None
            assert "known_models" in body["training_range"]


# ---------------------------------------------------------------------------
# Schema endpoint
# ---------------------------------------------------------------------------


class TestSchema:
    def test_schema_returns_200(self):
        response = client.get("/schema")
        assert response.status_code == 200

    def test_schema_has_required_columns(self):
        response = client.get("/schema")
        assert "required_columns" in response.json()
        assert isinstance(response.json()["required_columns"], list)

    def test_schema_has_min_row_count(self):
        response = client.get("/schema")
        assert response.json()["min_row_count"] == 100

    def test_schema_required_columns_count(self):
        response = client.get("/schema")
        assert len(response.json()["required_columns"]) == 7


# ---------------------------------------------------------------------------
# Profile endpoint
# ---------------------------------------------------------------------------


class TestProfile:
    def test_profile_returns_200(self):
        response = client.get("/predict/profile")
        assert response.status_code == 200

    def test_profile_has_cpu_cores(self):
        response = client.get("/predict/profile")
        assert "cpu_cores" in response.json()

    def test_profile_has_memory(self):
        response = client.get("/predict/profile")
        assert "memory_total_gb" in response.json()

    def test_profile_has_disk_type(self):
        response = client.get("/predict/profile")
        assert "disk_type" in response.json()


# ---------------------------------------------------------------------------
# Predict endpoint — validation
# ---------------------------------------------------------------------------


class TestPredictValidation:
    def test_predict_missing_model_name(self, valid_predict_payload):
        payload = {k: v for k, v in valid_predict_payload.items() if k != "model_name"}
        response = client.post("/predict", json=payload)
        assert response.status_code == 422

    def test_predict_missing_complexity_level(self, valid_predict_payload):
        payload = {k: v for k, v in valid_predict_payload.items() if k != "complexity_level"}
        response = client.post("/predict", json=payload)
        assert response.status_code == 422

    def test_predict_missing_cpu_count(self, valid_predict_payload):
        payload = {k: v for k, v in valid_predict_payload.items() if k != "cpu_count"}
        response = client.post("/predict", json=payload)
        assert response.status_code == 422

    def test_predict_invalid_model_name(self, valid_predict_payload):
        payload = {**valid_predict_payload, "model_name": "bert-large"}
        response = client.post("/predict", json=payload)
        assert response.status_code == 422

    def test_predict_complexity_too_high(self, valid_predict_payload):
        payload = {**valid_predict_payload, "complexity_level": 6}
        response = client.post("/predict", json=payload)
        assert response.status_code == 422

    def test_predict_complexity_too_low(self, valid_predict_payload):
        payload = {**valid_predict_payload, "complexity_level": 0}
        response = client.post("/predict", json=payload)
        assert response.status_code == 422

    def test_predict_negative_cpu_count(self, valid_predict_payload):
        payload = {**valid_predict_payload, "cpu_count": -1}
        response = client.post("/predict", json=payload)
        assert response.status_code == 422

    def test_predict_negative_total_memory_mb(self, valid_predict_payload):
        payload = {**valid_predict_payload, "total_memory_mb": -1.0}
        response = client.post("/predict", json=payload)
        assert response.status_code == 422


# ---------------------------------------------------------------------------
# Predict endpoint — model not available
# ---------------------------------------------------------------------------


class TestPredictNoModel:
    def test_predict_no_model_returns_503(self, valid_predict_payload, monkeypatch, tmp_path):
        monkeypatch.setattr("cetp.predictor.BASE_MODEL_PATH", tmp_path / "missing_base.pkl")
        monkeypatch.setattr("cetp.predictor.CUSTOM_MODEL_PATH", tmp_path / "missing_custom.pkl")
        response = client.post("/predict", json=valid_predict_payload)
        assert response.status_code == 503

    def test_predict_no_model_error_message(self, valid_predict_payload, monkeypatch, tmp_path):
        monkeypatch.setattr("cetp.predictor.BASE_MODEL_PATH", tmp_path / "missing_base.pkl")
        monkeypatch.setattr("cetp.predictor.CUSTOM_MODEL_PATH", tmp_path / "missing_custom.pkl")
        response = client.post("/predict", json=valid_predict_payload)
        assert "error" in response.json()

    def test_predict_no_model_has_detail(self, valid_predict_payload, monkeypatch, tmp_path):
        monkeypatch.setattr("cetp.predictor.BASE_MODEL_PATH", tmp_path / "missing_base.pkl")
        monkeypatch.setattr("cetp.predictor.CUSTOM_MODEL_PATH", tmp_path / "missing_custom.pkl")
        response = client.post("/predict", json=valid_predict_payload)
        body = response.json()
        assert "detail" in body
        assert len(body["detail"]) > 10


# ---------------------------------------------------------------------------
# Predict endpoint — valid request, real base model
# ---------------------------------------------------------------------------


class TestPredictValidPayload:
    def test_predict_valid_payload_returns_200(self, valid_predict_payload):
        response = client.post("/predict", json=valid_predict_payload)
        assert response.status_code == 200

    def test_predict_response_has_all_fields(self, valid_predict_payload):
        response = client.post("/predict", json=valid_predict_payload)
        body = response.json()
        expected_keys = {
            "predicted_runtime_sec",
            "confidence_interval",
            "confidence_status",
            "confidence_warnings",
            "sla_flag",
            "sla_threshold_sec",
            "top_shap_features",
            "model_artifact",
            "model_used",
            "complexity_level",
        }
        assert expected_keys.issubset(body.keys())

    def test_predict_model_used_matches_request(self, valid_predict_payload):
        """Regression test: the reported model must equal the model actually
        predicted, never a different one-hot category leaking through."""
        response = client.post("/predict", json=valid_predict_payload)
        body = response.json()
        assert body["model_used"] == valid_predict_payload["model_name"]

    def test_predict_top_shap_features_model_feature_matches_request(self, valid_predict_payload):
        """Regression test: if a 'model=X' SHAP feature is surfaced, X must be
        the model actually predicted — never a different one-hot category."""
        response = client.post("/predict", json=valid_predict_payload)
        body = response.json()
        model_features = [
            f for f in body["top_shap_features"] if f["feature"].startswith("cat__model_")
        ]
        for f in model_features:
            assert f["feature"] == f"cat__model_{valid_predict_payload['model_name']}"

    def test_predict_other_models_accepted(self, valid_predict_payload):
        for model_name in ("resnet50", "mobilenet", "distilbert"):
            payload = {**valid_predict_payload, "model_name": model_name}
            response = client.post("/predict", json=payload)
            assert response.status_code == 200
            assert response.json()["model_used"] == model_name

    def test_predict_in_range_inputs_are_reliable(self, valid_predict_payload):
        response = client.post("/predict", json=valid_predict_payload)
        assert response.json()["confidence_status"] == "RELIABLE"

    def test_predict_out_of_range_cpu_count_is_extrapolated(self, valid_predict_payload):
        payload = {**valid_predict_payload, "cpu_count": 128}
        response = client.post("/predict", json=payload)
        body = response.json()
        assert body["confidence_status"] == "EXTRAPOLATED"
        assert len(body["confidence_warnings"]) > 0

    def test_predict_sla_flag_is_valid_value(self, valid_predict_payload):
        response = client.post("/predict", json=valid_predict_payload)
        assert response.json()["sla_flag"] in {"GREEN", "YELLOW", "RED", "UNKNOWN"}

    def test_predict_top_shap_features_capped_at_three(self, valid_predict_payload):
        response = client.post("/predict", json=valid_predict_payload)
        assert len(response.json()["top_shap_features"]) <= 3

    def test_predict_custom_sla_config_overrides_default(self, valid_predict_payload):
        payload = {
            **valid_predict_payload,
            "sla_config": {"resnet18": {"sla_runtime_sec": 0.001, "warn_at_sec": 0.0001}},
        }
        response = client.post("/predict", json=payload)
        body = response.json()
        assert response.status_code == 200
        assert body["sla_flag"] == "RED"
        assert body["sla_threshold_sec"] == 0.001

    def test_predict_unconfigured_model_sla_is_unknown(self, valid_predict_payload):
        payload = {**valid_predict_payload, "sla_config": {}}
        response = client.post("/predict", json=payload)
        body = response.json()
        assert body["sla_flag"] == "UNKNOWN"
        assert body["sla_threshold_sec"] is None


# ---------------------------------------------------------------------------
# Explain endpoint
# ---------------------------------------------------------------------------


class TestExplain:
    def test_explain_no_model_returns_503(self, valid_predict_payload, monkeypatch, tmp_path):
        monkeypatch.setattr("cetp.predictor.BASE_MODEL_PATH", tmp_path / "missing_base.pkl")
        monkeypatch.setattr("cetp.predictor.CUSTOM_MODEL_PATH", tmp_path / "missing_custom.pkl")
        response = client.post("/explain", json=valid_predict_payload)
        assert response.status_code == 503

    def test_explain_missing_required_fields_returns_422(self):
        response = client.post("/explain", json={"model_name": "resnet18"})
        assert response.status_code == 422

    def test_explain_valid_payload_returns_200(self, valid_predict_payload):
        response = client.post("/explain", json=valid_predict_payload)
        assert response.status_code == 200

    def test_explain_response_has_all_fields(self, valid_predict_payload):
        response = client.post("/explain", json=valid_predict_payload)
        body = response.json()
        expected_keys = {
            "base_value",
            "predicted_runtime_sec",
            "confidence_status",
            "features",
            "model_used",
            "complexity_level",
        }
        assert expected_keys.issubset(body.keys())

    def test_explain_base_value_plus_features_reconstructs_prediction(self, valid_predict_payload):
        """Real correctness check for the whole SHAP pipeline as exposed over
        the API: base_value + sum(all feature shap_values) must reconstruct
        the actual predicted runtime, within a small floating-point tolerance."""
        response = client.post("/explain", json=valid_predict_payload)
        body = response.json()
        reconstructed = body["base_value"] + sum(f["shap_value"] for f in body["features"])
        assert abs(reconstructed - body["predicted_runtime_sec"]) < 1.0

    def test_explain_returns_more_features_than_predict_top_shap(self, valid_predict_payload):
        predict_response = client.post("/predict", json=valid_predict_payload)
        explain_response = client.post("/explain", json=valid_predict_payload)
        assert len(explain_response.json()["features"]) >= len(
            predict_response.json()["top_shap_features"]
        )

    def test_explain_model_feature_matches_request(self, valid_predict_payload):
        """Regression test: same one-hot aggregation fix as /predict — a
        'model=X' feature must never name a different model than requested."""
        response = client.post("/explain", json=valid_predict_payload)
        body = response.json()
        model_features = [f for f in body["features"] if f["feature"].startswith("cat__model_")]
        assert len(model_features) <= 1
        for f in model_features:
            assert f["feature"] == f"cat__model_{valid_predict_payload['model_name']}"


# ---------------------------------------------------------------------------
# Error handling
# ---------------------------------------------------------------------------


class TestErrorHandling:
    def test_404_returns_json(self):
        response = client.get("/nonexistent")
        assert response.status_code == 404
        assert response.headers["content-type"].startswith("application/json")

    def test_404_has_error_field(self):
        response = client.get("/nonexistent")
        assert "error" in response.json()

    def test_invalid_json_body_returns_422(self):
        response = client.post(
            "/predict",
            content=b"this is not json",
            headers={"Content-Type": "application/json"},
        )
        assert response.status_code == 422
