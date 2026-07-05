"""Unit tests for cetp.predictor."""

import time

import pytest

from cetp.predictor import RESIDUAL_STD, CETPPredictor


# ---------------------------------------------------------------------------
# Instantiation / load_model
# ---------------------------------------------------------------------------


class TestLoadModel:
    def test_loads_model_successfully_when_base_model_exists(self):
        """CETPPredictor() succeeds and populates a usable pipeline against the
        real package artefact at model/artifacts/base_model.pkl."""
        predictor = CETPPredictor()
        assert predictor.pipeline is not None
        assert hasattr(predictor.pipeline, "predict")

    def test_raises_file_not_found_when_no_model_exists(self, tmp_path, monkeypatch):
        """Construction raises FileNotFoundError with an actionable message when
        neither a custom nor a base model artefact can be found."""
        monkeypatch.setattr("cetp.predictor.BASE_MODEL_PATH", tmp_path / "missing_model.pkl")
        monkeypatch.setattr(
            "cetp.predictor.CUSTOM_MODEL_PATH", tmp_path / "no_home" / "custom_model.pkl"
        )

        with pytest.raises(FileNotFoundError) as exc_info:
            CETPPredictor()

        message = str(exc_info.value)
        assert "model artefact" in message
        assert "missing_model.pkl" in message

    def test_model_loads_in_under_two_seconds(self):
        """load_model() must complete quickly enough for interactive CLI use."""
        predictor = CETPPredictor()
        start = time.perf_counter()
        predictor.load_model()
        elapsed = time.perf_counter() - start
        assert elapsed < 2.0

    def test_explicit_model_path_is_used_when_given(self):
        """An explicit model_path takes priority and is loaded directly."""
        from cetp.predictor import BASE_MODEL_PATH

        predictor = CETPPredictor(model_path=str(BASE_MODEL_PATH))
        assert predictor.pipeline is not None

    def test_explicit_model_path_raises_when_missing(self, tmp_path):
        """An explicit but nonexistent model_path raises immediately, without
        falling back to the custom or base model."""
        with pytest.raises(FileNotFoundError, match="does not exist"):
            CETPPredictor(model_path=str(tmp_path / "nope.pkl"))

    def test_custom_model_path_takes_priority_over_base(self, tmp_path, monkeypatch):
        """A model present at CUSTOM_MODEL_PATH is preferred over the base model."""
        import shutil

        from cetp.predictor import BASE_MODEL_PATH

        custom_dir = tmp_path / "cetp_home"
        custom_dir.mkdir()
        custom_model = custom_dir / "custom_model.pkl"
        shutil.copy(BASE_MODEL_PATH, custom_model)

        monkeypatch.setattr("cetp.predictor.CUSTOM_MODEL_PATH", custom_model)

        predictor = CETPPredictor()
        assert predictor._model_path == custom_model


# ---------------------------------------------------------------------------
# predict()
# ---------------------------------------------------------------------------


class TestPredict:
    def test_predict_returns_all_required_keys(self):
        """A valid resnet18 complexity-3 request returns the full result dict shape."""
        predictor = CETPPredictor()
        result = predictor.predict("resnet18", 3, cpu_count=4, total_memory_mb=8000.0)

        expected_keys = {
            "predicted_runtime_sec",
            "confidence_interval",
            "confidence_status",
            "confidence_warnings",
            "model_used",
            "complexity_level",
            "batch_size",
            "num_iterations",
        }
        assert expected_keys.issubset(result.keys())
        assert isinstance(result["predicted_runtime_sec"], float)
        assert isinstance(result["confidence_interval"], list)
        assert len(result["confidence_interval"]) == 2
        assert result["model_used"] == "resnet18"
        assert result["complexity_level"] == 3

    def test_predict_confidence_interval_is_centered_on_point_estimate(self):
        """The interval should be point_estimate +/- 1.645 * RESIDUAL_STD."""
        predictor = CETPPredictor()
        result = predictor.predict("resnet18", 3, cpu_count=4, total_memory_mb=8000.0)
        lower, upper = result["confidence_interval"]
        point = result["predicted_runtime_sec"]
        assert upper == pytest.approx(point + 1.645 * RESIDUAL_STD)
        assert lower == pytest.approx(max(0.0, point - 1.645 * RESIDUAL_STD))

    def test_predict_raises_value_error_for_invalid_model_name(self):
        """An unknown model name raises ValueError, not a silent failure."""
        predictor = CETPPredictor()
        with pytest.raises(ValueError, match="bert-large"):
            predictor.predict("bert-large", 3, cpu_count=4, total_memory_mb=8000.0)

    def test_predict_raises_value_error_for_complexity_zero(self):
        """complexity_level=0 is out of range and raises ValueError."""
        predictor = CETPPredictor()
        with pytest.raises(ValueError):
            predictor.predict("resnet18", 0, cpu_count=4, total_memory_mb=8000.0)

    def test_predict_raises_value_error_for_complexity_six(self):
        """complexity_level=6 is out of range and raises ValueError."""
        predictor = CETPPredictor()
        with pytest.raises(ValueError):
            predictor.predict("resnet18", 6, cpu_count=4, total_memory_mb=8000.0)

    def test_predict_wraps_unexpected_pipeline_failure_in_runtime_error(self, monkeypatch):
        """A failure inside the pipeline itself (post-validation) is never a raw
        exception — it's wrapped in a RuntimeError with context."""
        predictor = CETPPredictor()

        def _boom(*_args, **_kwargs):
            raise RuntimeError("pipeline exploded")

        monkeypatch.setattr(predictor.pipeline, "predict", _boom)

        with pytest.raises(RuntimeError, match="Prediction failed"):
            predictor.predict("resnet18", 3, cpu_count=4, total_memory_mb=8000.0)


# ---------------------------------------------------------------------------
# check_confidence()
# ---------------------------------------------------------------------------


class TestCheckConfidence:
    def test_reliable_for_in_range_inputs(self):
        """cpu_count=4, total_memory_mb=8000 are within the trained range."""
        predictor = CETPPredictor()
        result = predictor.check_confidence(
            cpu_count=4, total_memory_mb=8000.0, model_name="resnet18"
        )
        assert result["status"] == "RELIABLE"
        assert result["warnings"] == []

    def test_extrapolated_for_out_of_range_cpu_count(self):
        """cpu_count=16 is well outside the trained 2-4 range."""
        predictor = CETPPredictor()
        result = predictor.check_confidence(
            cpu_count=16, total_memory_mb=8000.0, model_name="resnet18"
        )
        assert result["status"] == "EXTRAPOLATED"
        assert any("cpu_count" in w for w in result["warnings"])

    def test_extrapolated_for_out_of_range_memory(self):
        """total_memory_mb=32000 is well outside the trained range."""
        predictor = CETPPredictor()
        result = predictor.check_confidence(
            cpu_count=4, total_memory_mb=32000.0, model_name="resnet18"
        )
        assert result["status"] == "EXTRAPOLATED"
        assert any("total_memory_mb" in w for w in result["warnings"])

    def test_extrapolated_for_unknown_model(self):
        """A model not present in known_models triggers EXTRAPOLATED."""
        predictor = CETPPredictor()
        result = predictor.check_confidence(
            cpu_count=4, total_memory_mb=8000.0, model_name="not_a_known_model"
        )
        assert result["status"] == "EXTRAPOLATED"
        assert any("not_a_known_model" in w for w in result["warnings"])


# ---------------------------------------------------------------------------
# compute_sla_flag()
# ---------------------------------------------------------------------------


class TestComputeSlaFlag:
    def test_green_below_warn_threshold(self):
        predictor = CETPPredictor()
        thresholds = predictor._sla_config["resnet18"]
        flag = predictor.compute_sla_flag(thresholds["warn_at_sec"] - 1.0, "resnet18")
        assert flag == "GREEN"

    def test_green_exactly_at_warn_threshold(self):
        """warn_at_sec is inclusive of GREEN."""
        predictor = CETPPredictor()
        thresholds = predictor._sla_config["resnet18"]
        flag = predictor.compute_sla_flag(thresholds["warn_at_sec"], "resnet18")
        assert flag == "GREEN"

    def test_yellow_between_warn_and_sla(self):
        predictor = CETPPredictor()
        thresholds = predictor._sla_config["resnet18"]
        midpoint = (thresholds["warn_at_sec"] + thresholds["sla_runtime_sec"]) / 2
        flag = predictor.compute_sla_flag(midpoint, "resnet18")
        assert flag == "YELLOW"

    def test_yellow_exactly_at_sla_threshold(self):
        """sla_runtime_sec is inclusive of YELLOW, not RED."""
        predictor = CETPPredictor()
        thresholds = predictor._sla_config["resnet18"]
        flag = predictor.compute_sla_flag(thresholds["sla_runtime_sec"], "resnet18")
        assert flag == "YELLOW"

    def test_red_above_sla_threshold(self):
        predictor = CETPPredictor()
        thresholds = predictor._sla_config["resnet18"]
        flag = predictor.compute_sla_flag(thresholds["sla_runtime_sec"] + 1.0, "resnet18")
        assert flag == "RED"

    def test_raises_key_error_for_unknown_model(self):
        predictor = CETPPredictor()
        with pytest.raises(KeyError):
            predictor.compute_sla_flag(100.0, "not_a_configured_model")


# ---------------------------------------------------------------------------
# get_shap_explanation()
# ---------------------------------------------------------------------------


class TestGetShapExplanation:
    def test_returns_dict_with_base_value_and_features(self):
        predictor = CETPPredictor()
        explanation = predictor.get_shap_explanation(
            "resnet18", 3, cpu_count=4, total_memory_mb=8000.0, top_n=3
        )
        assert set(explanation.keys()) == {"base_value", "features"}
        assert isinstance(explanation["base_value"], float)

    def test_returns_top_n_items_with_correct_keys(self):
        predictor = CETPPredictor()
        explanation = predictor.get_shap_explanation(
            "resnet18", 3, cpu_count=4, total_memory_mb=8000.0, top_n=3
        )
        features = explanation["features"]
        assert len(features) == 3
        for item in features:
            assert set(item.keys()) == {"feature", "shap_value", "direction"}
            assert isinstance(item["feature"], str)
            assert isinstance(item["shap_value"], float)
            assert item["direction"] in {"increases_runtime", "decreases_runtime"}

    def test_respects_custom_top_n(self):
        predictor = CETPPredictor()
        explanation = predictor.get_shap_explanation(
            "resnet18", 3, cpu_count=4, total_memory_mb=8000.0, top_n=2
        )
        assert len(explanation["features"]) == 2

    def test_sorted_by_absolute_shap_value_descending(self):
        predictor = CETPPredictor()
        explanation = predictor.get_shap_explanation(
            "resnet18", 3, cpu_count=4, total_memory_mb=8000.0, top_n=5
        )
        magnitudes = [abs(item["shap_value"]) for item in explanation["features"]]
        assert magnitudes == sorted(magnitudes, reverse=True)

    def test_model_feature_never_names_a_different_model(self):
        """Regression test: one-hot encoding splits 'model' into 4 binary
        columns, and SHAP assigns a value to every column — including the
        OFF ones. A prior bug ranked raw one-hot columns individually, so
        predicting resnet18 could surface 'cat__model_distilbert' as a top
        contributor purely because its OFF-column SHAP value happened to be
        large, which is incoherent to show a user. If a 'model=X' feature
        appears in the top-N at all, X must be the model actually predicted."""
        predictor = CETPPredictor()
        explanation = predictor.get_shap_explanation(
            "resnet18", 3, cpu_count=4, total_memory_mb=8000.0, top_n=3
        )
        model_features = [
            item for item in explanation["features"] if item["feature"].startswith("cat__model_")
        ]
        assert len(model_features) <= 1, "at most one grouped 'model' feature should ever appear"
        for item in model_features:
            assert item["feature"] == "cat__model_resnet18"

    def test_base_value_plus_shap_values_reconstructs_prediction(self):
        """The real mathematical correctness check for the whole SHAP
        pipeline: base_value + sum(all feature shap_values) — not just the
        top-N — must reconstruct the model's actual raw prediction, within a
        small tolerance for floating point and the one-hot aggregation from
        the earlier fix. top_n=100 guarantees every post-aggregation feature
        (cpu_count, total_memory_mb, complexity_level, batch_size,
        num_iterations, and the aggregated model feature) is included, not
        just the top 3."""
        predictor = CETPPredictor()
        prediction = predictor.predict("resnet18", 3, cpu_count=4, total_memory_mb=8000.0)
        explanation = predictor.get_shap_explanation(
            "resnet18", 3, cpu_count=4, total_memory_mb=8000.0, top_n=100
        )

        reconstructed = explanation["base_value"] + sum(
            item["shap_value"] for item in explanation["features"]
        )
        actual = prediction["predicted_runtime_sec"]

        assert abs(reconstructed - actual) < 1.0

    def test_returns_empty_features_on_internal_failure(self, monkeypatch):
        """A broken preprocessor step must degrade to empty features, never raise."""
        predictor = CETPPredictor()

        def _boom(*_args, **_kwargs):
            raise RuntimeError("boom")

        preprocessor = predictor.pipeline.named_steps["prep"]
        monkeypatch.setattr(preprocessor, "transform", _boom)

        explanation = predictor.get_shap_explanation(
            "resnet18", 3, cpu_count=4, total_memory_mb=8000.0
        )
        assert explanation == {"base_value": 0.0, "features": []}
