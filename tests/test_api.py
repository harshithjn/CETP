"""Comprehensive tests for the CETP FastAPI REST backend (Phase 7)."""
import pytest
from fastapi.testclient import TestClient

from api.main import app, compute_derived_features

client = TestClient(app)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def valid_predict_payload():
    return {
        "workload_type": "ML",
        "workload_name": "ml_resnet",
        "workload_complexity": 3,
        "cpu_cores": 8,
        "memory_total_gb": 16.0,
        "cpu_avg_pct": 45.0,
        "memory_avg_gb": 6.4,
        "disk_read_mb": 120.5,
        "disk_write_mb": 34.2,
        "disk_type": "SSD",
        "disk_speed_class": 2,
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
        assert response.json()["min_row_count"] == 500

    def test_schema_required_columns_count(self):
        response = client.get("/schema")
        assert len(response.json()["required_columns"]) == 16


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
    def test_predict_missing_workload_type(self, valid_predict_payload):
        payload = {k: v for k, v in valid_predict_payload.items() if k != "workload_type"}
        response = client.post("/predict", json=payload)
        assert response.status_code == 422

    def test_predict_missing_workload_name(self, valid_predict_payload):
        payload = {k: v for k, v in valid_predict_payload.items() if k != "workload_name"}
        response = client.post("/predict", json=payload)
        assert response.status_code == 422

    def test_predict_missing_cpu_cores(self, valid_predict_payload):
        payload = {k: v for k, v in valid_predict_payload.items() if k != "cpu_cores"}
        response = client.post("/predict", json=payload)
        assert response.status_code == 422

    def test_predict_invalid_workload_type(self, valid_predict_payload):
        payload = {**valid_predict_payload, "workload_type": "INVALID"}
        response = client.post("/predict", json=payload)
        assert response.status_code == 422

    def test_predict_invalid_disk_type(self, valid_predict_payload):
        payload = {**valid_predict_payload, "disk_type": "OPTANE"}
        response = client.post("/predict", json=payload)
        assert response.status_code == 422

    def test_predict_complexity_too_high(self, valid_predict_payload):
        payload = {**valid_predict_payload, "workload_complexity": 6}
        response = client.post("/predict", json=payload)
        assert response.status_code == 422

    def test_predict_complexity_too_low(self, valid_predict_payload):
        payload = {**valid_predict_payload, "workload_complexity": 0}
        response = client.post("/predict", json=payload)
        assert response.status_code == 422

    def test_predict_negative_cpu_cores(self, valid_predict_payload):
        payload = {**valid_predict_payload, "cpu_cores": -1}
        response = client.post("/predict", json=payload)
        assert response.status_code == 422

    def test_predict_cpu_pct_over_100(self, valid_predict_payload):
        payload = {**valid_predict_payload, "cpu_avg_pct": 101}
        response = client.post("/predict", json=payload)
        assert response.status_code == 422


# ---------------------------------------------------------------------------
# Predict endpoint — model not available
# ---------------------------------------------------------------------------

class TestPredictNoModel:
    def test_predict_no_model_returns_503(self, valid_predict_payload):
        response = client.post("/predict", json=valid_predict_payload)
        assert response.status_code == 503

    def test_predict_no_model_error_message(self, valid_predict_payload):
        response = client.post("/predict", json=valid_predict_payload)
        assert "error" in response.json()

    def test_predict_no_model_has_detail(self, valid_predict_payload):
        response = client.post("/predict", json=valid_predict_payload)
        body = response.json()
        assert "detail" in body
        assert len(body["detail"]) > 10


# ---------------------------------------------------------------------------
# Predict endpoint — valid request structure
# ---------------------------------------------------------------------------

class TestPredictValidPayload:
    def test_predict_valid_payload_accepted(self, valid_predict_payload):
        response = client.post("/predict", json=valid_predict_payload)
        assert response.status_code in (200, 503)

    def test_predict_db_workload_accepted(self, valid_predict_payload):
        payload = {**valid_predict_payload, "workload_type": "DB"}
        response = client.post("/predict", json=payload)
        assert response.status_code in (200, 503)

    def test_predict_web_workload_accepted(self, valid_predict_payload):
        payload = {**valid_predict_payload, "workload_type": "WEB"}
        response = client.post("/predict", json=payload)
        assert response.status_code in (200, 503)

    def test_predict_optional_fields_have_defaults(self, valid_predict_payload):
        payload = {k: v for k, v in valid_predict_payload.items() if k != "disk_read_mb"}
        response = client.post("/predict", json=payload)
        assert response.status_code in (200, 503)


# ---------------------------------------------------------------------------
# Explain endpoint
# ---------------------------------------------------------------------------

class TestExplain:
    def test_explain_no_model_returns_503(self, valid_predict_payload):
        response = client.post("/explain", json=valid_predict_payload)
        assert response.status_code == 503

    def test_explain_missing_required_fields_returns_422(self):
        response = client.post("/explain", json={"workload_type": "ML"})
        assert response.status_code == 422

    def test_explain_valid_payload_accepted(self, valid_predict_payload):
        response = client.post("/explain", json=valid_predict_payload)
        assert response.status_code in (200, 503)


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


# ---------------------------------------------------------------------------
# compute_derived_features unit tests
# ---------------------------------------------------------------------------

class TestComputeDerivedFeatures:
    def _base(self):
        return {
            "cpu_cores": 8,
            "cpu_avg_pct": 25.0,
            "memory_avg_gb": 4.0,
            "memory_total_gb": 8.0,
            "disk_read_mb": 100.0,
            "disk_write_mb": 50.0,
        }

    def test_compute_effective_cpu(self):
        result = compute_derived_features(self._base())
        assert result["effective_cpu"] == 6.0

    def test_compute_memory_pressure(self):
        result = compute_derived_features(self._base())
        assert result["memory_pressure"] == 0.5

    def test_compute_io_intensity_defaults_to_zero_without_runtime(self):
        data = self._base()  # no runtime_sec key
        result = compute_derived_features(data)
        assert result["io_intensity"] == 0.0
