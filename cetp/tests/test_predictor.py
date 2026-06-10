"""Unit tests for cetp.predictor (structural tests — no model artefact required)."""
import pytest

from cetp.predictor import CETPPredictor, SLA_FLAGS


# ---------------------------------------------------------------------------
# Instantiation
# ---------------------------------------------------------------------------

class TestCETPPredictorInstantiation:
    def test_instantiation_succeeds_without_model_file(self):
        """CETPPredictor() must not raise even when no model artefact is present."""
        predictor = CETPPredictor()
        assert predictor is not None

    def test_instantiation_with_custom_sla_path(self, tmp_path):
        """CETPPredictor accepts a custom sla_config_path without raising."""
        sla_file = tmp_path / "custom_sla.json"
        sla_file.write_text(
            '{"ML": {"sla_runtime_sec": 30.0, "warn_at_sec": 20.0}}'
        )
        predictor = CETPPredictor(sla_config_path=str(sla_file))
        assert predictor is not None

    def test_instantiation_with_nonexistent_sla_path_does_not_raise(self):
        """A missing SLA file should not raise at __init__ time."""
        predictor = CETPPredictor(sla_config_path="/nonexistent/path/sla.json")
        assert predictor is not None


# ---------------------------------------------------------------------------
# load_model
# ---------------------------------------------------------------------------

class TestLoadModel:
    def test_raises_file_not_found_when_no_model_exists(self, tmp_path, monkeypatch):
        """load_model() must raise when neither base nor custom model is present."""
        # Point BASE_MODEL_PATH to a non-existent file
        monkeypatch.setattr("cetp.predictor.BASE_MODEL_PATH", tmp_path / "missing_model.pkl")
        monkeypatch.setattr("cetp.predictor.CETP_DIR", tmp_path / "no_cetp_dir")
        predictor = CETPPredictor()
        with pytest.raises((FileNotFoundError, NotImplementedError)):
            predictor.load_model()

    def test_load_model_raises_with_informative_message(self, tmp_path, monkeypatch):
        """The error message from load_model() must mention what file is needed."""
        monkeypatch.setattr("cetp.predictor.BASE_MODEL_PATH", tmp_path / "missing.pkl")
        monkeypatch.setattr("cetp.predictor.CETP_DIR", tmp_path / "no_dir")
        predictor = CETPPredictor()
        with pytest.raises((FileNotFoundError, NotImplementedError)) as exc_info:
            predictor.load_model()
        # Message should mention the model file
        assert "model" in str(exc_info.value).lower()


# ---------------------------------------------------------------------------
# get_active_model_info
# ---------------------------------------------------------------------------

class TestGetActiveModelInfo:
    def test_returns_dict_with_expected_keys(self):
        """get_active_model_info() must return all required keys without a loaded model."""
        predictor = CETPPredictor()
        info = predictor.get_active_model_info()
        required_keys = {"model_path", "model_version", "trained_on", "feature_count", "custom_model"}
        assert required_keys.issubset(set(info.keys())), (
            f"Missing keys: {required_keys - set(info.keys())}"
        )

    def test_model_path_is_string(self):
        """model_path in the info dict must be a string."""
        predictor = CETPPredictor()
        info = predictor.get_active_model_info()
        assert isinstance(info["model_path"], str)

    def test_custom_model_is_bool(self):
        """custom_model in the info dict must be a bool."""
        predictor = CETPPredictor()
        info = predictor.get_active_model_info()
        assert isinstance(info["custom_model"], bool)

    def test_feature_count_is_int(self):
        """feature_count in the info dict must be an int."""
        predictor = CETPPredictor()
        info = predictor.get_active_model_info()
        assert isinstance(info["feature_count"], int)


# ---------------------------------------------------------------------------
# SLA_FLAGS constant
# ---------------------------------------------------------------------------

class TestSLAFlags:
    def test_sla_flags_contains_expected_values(self):
        assert "GREEN" in SLA_FLAGS
        assert "YELLOW" in SLA_FLAGS
        assert "RED" in SLA_FLAGS

    def test_sla_flags_has_three_elements(self):
        assert len(SLA_FLAGS) == 3
