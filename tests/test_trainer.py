"""Unit tests for cetp.trainer."""

import csv
import json

import numpy as np
import pytest

from cetp.trainer import CETPTrainer, LowAccuracyWarning
from cetp.validator import REQUIRED_COLUMNS, InsufficientDataError, ValidationError

_MODELS = ["resnet18", "resnet50", "mobilenet", "distilbert"]
_MODEL_OFFSET = {"resnet18": 0.0, "resnet50": 50.0, "mobilenet": 30.0, "distilbert": 80.0}


def make_training_csv(
    path, n_rows: int = 150, scramble_runtime: bool = False, seed: int = 42
) -> str:
    """
    Writes a synthetic CSV matching the current CETP schema. runtime_sec is a
    rough (noisy) function of the other columns so a model can actually learn
    something — not pure random noise — unless scramble_runtime shuffles it
    to deliberately destroy that relationship (for the low-accuracy test).
    """
    rng = np.random.default_rng(seed)

    cpu_count = rng.integers(2, 9, size=n_rows)
    total_memory_mb = rng.uniform(2000.0, 16000.0, size=n_rows)
    model = rng.choice(_MODELS, size=n_rows)
    complexity_level = rng.integers(1, 6, size=n_rows)
    batch_size = rng.integers(8, 129, size=n_rows)
    num_iterations = rng.integers(19, 974, size=n_rows)

    base = (
        num_iterations * 0.3
        + complexity_level * 10.0
        - cpu_count * 5.0
        + batch_size * 0.5
        + np.array([_MODEL_OFFSET[m] for m in model])
    )
    noise = rng.normal(0, base.std() * 0.05 + 1.0, size=n_rows)
    runtime_sec = np.clip(base + noise, 0.5, None)

    if scramble_runtime:
        runtime_sec = rng.permutation(runtime_sec)

    rows = []
    for i in range(n_rows):
        rows.append(
            {
                "cpu_count": int(cpu_count[i]),
                "total_memory_mb": float(total_memory_mb[i]),
                "model": model[i],
                "complexity_level": int(complexity_level[i]),
                "batch_size": int(batch_size[i]),
                "num_iterations": int(num_iterations[i]),
                "runtime_sec": float(runtime_sec[i]),
            }
        )

    path = str(path)
    with open(path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=REQUIRED_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)

    return path


# ---------------------------------------------------------------------------
# Basic training run
# ---------------------------------------------------------------------------


class TestTrainCompletes:
    def test_train_completes_without_error(self, tmp_path):
        """train() runs end-to-end on a valid synthetic CSV and returns a dict."""
        csv_path = make_training_csv(tmp_path / "train.csv")
        trainer = CETPTrainer(output_dir=str(tmp_path / "out"))
        result = trainer.train(csv_path)
        assert isinstance(result, dict)
        assert result["row_count"] == 150

    def test_train_result_has_expected_keys(self, tmp_path):
        csv_path = make_training_csv(tmp_path / "train.csv")
        trainer = CETPTrainer(output_dir=str(tmp_path / "out"))
        result = trainer.train(csv_path)
        expected_keys = {
            "model_path",
            "training_range_path",
            "metadata_path",
            "row_count",
            "metrics",
            "hyperparameters",
            "low_accuracy_warning",
        }
        assert expected_keys.issubset(result.keys())

    def test_train_result_echoes_base_hyperparams(self, tmp_path):
        """The returned dict's 'hyperparameters' key echoes BASE_HYPERPARAMS.
        This alone does NOT prove the fitted XGBRegressor actually used them —
        see test_trained_xgb_actually_used_base_hyperparams for that proof."""
        from cetp.trainer import BASE_HYPERPARAMS

        csv_path = make_training_csv(tmp_path / "train.csv")
        trainer = CETPTrainer(output_dir=str(tmp_path / "out"))
        result = trainer.train(csv_path)
        assert result["hyperparameters"] == BASE_HYPERPARAMS

    def test_trained_xgb_actually_used_base_hyperparams(self, tmp_path):
        """Reload the persisted pipeline and inspect the FITTED XGBRegressor's
        own get_params() — proof the warm-start hyperparameters were actually
        passed into the estimator, not just echoed in the result dict.
        XGBRegressor() with no args reports n_estimators/max_depth/learning_rate/
        random_state as None (verified separately), so a match here can only
        happen if BASE_HYPERPARAMS was really used to construct the model.
        """
        import joblib

        from cetp.trainer import BASE_HYPERPARAMS

        csv_path = make_training_csv(tmp_path / "train.csv")
        trainer = CETPTrainer(output_dir=str(tmp_path / "out"))
        result = trainer.train(csv_path)

        pipe = joblib.load(result["model_path"])
        trained_xgb = pipe.named_steps["model"]
        params = trained_xgb.get_params()

        assert params["n_estimators"] == BASE_HYPERPARAMS["n_estimators"]
        assert params["max_depth"] == BASE_HYPERPARAMS["max_depth"]
        assert params["learning_rate"] == BASE_HYPERPARAMS["learning_rate"]
        assert params["random_state"] == BASE_HYPERPARAMS["random_state"]


# ---------------------------------------------------------------------------
# Validation failure paths
# ---------------------------------------------------------------------------


class TestTrainValidationFailures:
    def test_train_raises_validation_error_for_missing_columns(self, tmp_path, capsys):
        """A CSV missing required columns raises ValidationError, not a crash."""
        cols = [c for c in REQUIRED_COLUMNS if c != "runtime_sec"]
        csv_path = str(tmp_path / "missing_col.csv")
        with open(csv_path, "w", newline="", encoding="utf-8") as fh:
            writer = csv.DictWriter(fh, fieldnames=cols)
            writer.writeheader()
            writer.writerows([{c: "1" for c in cols} for _ in range(150)])

        trainer = CETPTrainer(output_dir=str(tmp_path / "out"))
        with pytest.raises(ValidationError):
            trainer.train(csv_path)

        captured = capsys.readouterr()
        assert "runtime_sec" in captured.out

    def test_train_raises_insufficient_data_error(self, tmp_path):
        """Fewer than MIN_ROW_COUNT (100) rows raises InsufficientDataError."""
        csv_path = make_training_csv(tmp_path / "small.csv", n_rows=50)
        trainer = CETPTrainer(output_dir=str(tmp_path / "out"))
        with pytest.raises(InsufficientDataError):
            trainer.train(csv_path)

    def test_insufficient_data_error_is_a_validation_error(self, tmp_path):
        """InsufficientDataError must still be catchable as ValidationError."""
        csv_path = make_training_csv(tmp_path / "small.csv", n_rows=50)
        trainer = CETPTrainer(output_dir=str(tmp_path / "out"))
        with pytest.raises(ValidationError):
            trainer.train(csv_path)


# ---------------------------------------------------------------------------
# Output artefacts
# ---------------------------------------------------------------------------


class TestOutputArtefacts:
    def test_all_three_output_files_created_and_nonempty(self, tmp_path):
        csv_path = make_training_csv(tmp_path / "train.csv")
        out_dir = tmp_path / "out"
        trainer = CETPTrainer(output_dir=str(out_dir))
        trainer.train(csv_path)

        model_file = out_dir / "custom_model.pkl"
        range_file = out_dir / "custom_training_range.json"
        meta_file = out_dir / "custom_model_meta.json"

        for f in (model_file, range_file, meta_file):
            assert f.exists(), f"{f} was not created"
            assert f.stat().st_size > 0, f"{f} is empty"

    def test_training_range_has_expected_structure(self, tmp_path):
        csv_path = make_training_csv(tmp_path / "train.csv")
        out_dir = tmp_path / "out"
        trainer = CETPTrainer(output_dir=str(out_dir))
        trainer.train(csv_path)

        with open(out_dir / "custom_training_range.json", encoding="utf-8") as fh:
            training_range = json.load(fh)

        for key in ("cpu_count", "total_memory_mb", "batch_size", "num_iterations"):
            assert "min" in training_range[key]
            assert "max" in training_range[key]
        assert set(training_range["known_models"]) == set(_MODELS)

    def test_metadata_has_expected_structure(self, tmp_path):
        csv_path = make_training_csv(tmp_path / "train.csv")
        out_dir = tmp_path / "out"
        trainer = CETPTrainer(output_dir=str(out_dir))
        trainer.train(csv_path)

        with open(out_dir / "custom_model_meta.json", encoding="utf-8") as fh:
            metadata = json.load(fh)

        assert "trained_at" in metadata
        assert metadata["source_csv"] == csv_path
        assert metadata["row_count"] == 150
        assert set(metadata["metrics"].keys()) == {"mae", "rmse", "r2", "mape"}

    def test_output_dir_created_if_missing(self, tmp_path):
        """CETPTrainer's __init__ creates output_dir if it doesn't already exist."""
        out_dir = tmp_path / "does_not_exist_yet"
        assert not out_dir.exists()
        CETPTrainer(output_dir=str(out_dir))
        assert out_dir.exists()


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------


class TestMetrics:
    def test_metrics_dict_has_four_expected_keys(self, tmp_path):
        csv_path = make_training_csv(tmp_path / "train.csv")
        trainer = CETPTrainer(output_dir=str(tmp_path / "out"))
        result = trainer.train(csv_path)
        assert set(result["metrics"].keys()) == {"mae", "rmse", "r2", "mape"}
        for value in result["metrics"].values():
            assert isinstance(value, float)


# ---------------------------------------------------------------------------
# Low accuracy warning
# ---------------------------------------------------------------------------


class TestLowAccuracyWarning:
    def test_low_accuracy_warning_emitted_for_scrambled_runtime(self, tmp_path):
        """Scrambled runtime_sec destroys the learnable signal, forcing R² well
        below the 0.80 threshold, which must emit LowAccuracyWarning rather
        than raise an exception."""
        csv_path = make_training_csv(tmp_path / "scrambled.csv", scramble_runtime=True)
        trainer = CETPTrainer(output_dir=str(tmp_path / "out"))

        with pytest.warns(LowAccuracyWarning, match="LowAccuracy"):
            result = trainer.train(csv_path)

        assert result["low_accuracy_warning"] is True
        assert result["metrics"]["r2"] < 0.80

    def test_low_accuracy_warning_printed_not_raised(self, tmp_path, capsys):
        """The low-accuracy condition must not prevent train() from returning
        normally, and must be printed for CLI visibility."""
        csv_path = make_training_csv(tmp_path / "scrambled.csv", scramble_runtime=True)
        trainer = CETPTrainer(output_dir=str(tmp_path / "out"))

        with pytest.warns(LowAccuracyWarning, match="LowAccuracy"):
            result = trainer.train(csv_path)

        captured = capsys.readouterr()
        assert "LowAccuracyWarning" in captured.out
        assert result is not None

    def test_well_correlated_data_does_not_warn(self, tmp_path, recwarn):
        """A model that learns a real signal should clear the R² threshold and
        not trigger LowAccuracyWarning."""
        csv_path = make_training_csv(tmp_path / "train.csv")
        trainer = CETPTrainer(output_dir=str(tmp_path / "out"))
        result = trainer.train(csv_path)

        assert result["low_accuracy_warning"] is False
        assert not any(issubclass(w.category, LowAccuracyWarning) for w in recwarn.list)
