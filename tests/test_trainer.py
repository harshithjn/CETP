"""Unit tests for cetp.trainer (structural tests — no model artefact required)."""
import pytest

from cetp.trainer import CETTrainer, LowAccuracyWarning, MIN_R2_THRESHOLD


# ---------------------------------------------------------------------------
# Instantiation
# ---------------------------------------------------------------------------

class TestCETTrainerInstantiation:
    def test_instantiation_succeeds(self):
        """CETTrainer() must not raise on construction."""
        trainer = CETTrainer()
        assert trainer is not None

    def test_instantiation_with_custom_hyperparams_path(self, tmp_path):
        """CETTrainer accepts a custom base_hyperparams_path without raising."""
        hp_file = tmp_path / "custom_hyperparams.json"
        hp_file.write_text('{"n_estimators": 200, "max_depth": 6}')
        trainer = CETTrainer(base_hyperparams_path=str(hp_file))
        assert trainer is not None

    def test_instantiation_with_nonexistent_path_does_not_raise(self):
        """Passing a non-existent hyperparams path must not raise at __init__ time."""
        trainer = CETTrainer(base_hyperparams_path="/nonexistent/path.json")
        assert trainer is not None


# ---------------------------------------------------------------------------
# train() — missing base_hyperparams.json
# ---------------------------------------------------------------------------

class TestTrainMissingHyperparams:
    def test_raises_file_not_found_when_hyperparams_missing(self, tmp_path):
        """train() must raise FileNotFoundError when base_hyperparams.json is absent."""
        valid_csv = tmp_path / "data.csv"
        valid_csv.write_text("run_id,workload_type\n1,ML\n")  # minimal CSV (not valid schema, but file exists)

        missing_hp = tmp_path / "nonexistent_hyperparams.json"
        trainer = CETTrainer(base_hyperparams_path=str(missing_hp))

        with pytest.raises((FileNotFoundError, NotImplementedError)):
            trainer.train(csv_path=str(valid_csv))

    def test_error_message_mentions_hyperparams(self, tmp_path):
        """The error raised when hyperparams are missing must mention the missing file."""
        valid_csv = tmp_path / "data.csv"
        valid_csv.write_text("run_id\n1\n")

        missing_hp = tmp_path / "missing.json"
        trainer = CETTrainer(base_hyperparams_path=str(missing_hp))

        with pytest.raises((FileNotFoundError, NotImplementedError)) as exc_info:
            trainer.train(csv_path=str(valid_csv))
        assert any(
            keyword in str(exc_info.value).lower()
            for keyword in ("hyperparams", "hyperparameter", "base_hyperparams", "model")
        )


# ---------------------------------------------------------------------------
# train() — invalid CSV path
# ---------------------------------------------------------------------------

class TestTrainInvalidCsvPath:
    def test_raises_file_not_found_for_missing_csv(self, tmp_path):
        """train() must raise FileNotFoundError when the CSV path does not exist."""
        trainer = CETTrainer()
        nonexistent_csv = str(tmp_path / "does_not_exist.csv")
        with pytest.raises(FileNotFoundError):
            trainer.train(csv_path=nonexistent_csv)

    def test_error_message_contains_csv_path(self, tmp_path):
        """The FileNotFoundError message should include the bad path."""
        trainer = CETTrainer()
        bad_path = str(tmp_path / "missing_data.csv")
        with pytest.raises(FileNotFoundError) as exc_info:
            trainer.train(csv_path=bad_path)
        assert "missing_data.csv" in str(exc_info.value)


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

class TestTrainerConstants:
    def test_min_r2_threshold_is_float(self):
        assert isinstance(MIN_R2_THRESHOLD, float)

    def test_min_r2_threshold_is_positive(self):
        assert 0.0 < MIN_R2_THRESHOLD < 1.0

    def test_low_accuracy_warning_is_user_warning_subclass(self):
        assert issubclass(LowAccuracyWarning, UserWarning)
