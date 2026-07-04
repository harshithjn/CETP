"""Tests for cetp CLI — Phase 6."""

import csv
import json
import os

from click.testing import CliRunner

from cetp.cli import main
from cetp.validator import MIN_ROW_COUNT, REQUIRED_COLUMNS, VALID_MODELS


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_valid_csv(path: str, n_rows: int = 150) -> None:
    """Write a CETP-schema-valid CSV (current cpu_count/total_memory_mb/model/
    complexity_level/batch_size/num_iterations/runtime_sec schema) with
    n_rows data rows to path."""
    models = sorted(VALID_MODELS)
    with open(path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=REQUIRED_COLUMNS)
        writer.writeheader()
        for i in range(n_rows):
            writer.writerow(
                {
                    "cpu_count": str(2 + i % 4),
                    "total_memory_mb": f"{4000.0 + i * 10:.1f}",
                    "model": models[i % len(models)],
                    "complexity_level": str((i % 5) + 1),
                    "batch_size": str(8 * (1 + i % 4)),
                    "num_iterations": str(50 + i),
                    "runtime_sec": f"{10.0 + i * 0.1:.2f}",
                }
            )


def _make_schema_violation_csv(path: str, n_rows: int = 500) -> None:
    """Write a 500-row CSV where every row has an invalid workload_type (triggers ValidationError)."""
    fieldnames = [
        "run_id",
        "workload_type",
        "workload_name",
        "workload_complexity",
        "cpu_cores",
        "memory_total_gb",
        "cpu_avg_pct",
        "effective_cpu",
        "memory_avg_gb",
        "memory_pressure",
        "disk_read_mb",
        "disk_write_mb",
        "io_intensity",
        "disk_type",
        "disk_speed_class",
        "runtime_sec",
    ]
    with open(path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        for i in range(n_rows):
            writer.writerow(
                {
                    "run_id": str(i),
                    "workload_type": "INVALID",  # deliberate bad value
                    "workload_name": f"wl_{i}",
                    "workload_complexity": "3",
                    "cpu_cores": "8",
                    "memory_total_gb": "16.0",
                    "cpu_avg_pct": "45.0",
                    "effective_cpu": "4.4",
                    "memory_avg_gb": "6.0",
                    "memory_pressure": "0.375",
                    "disk_read_mb": "100.0",
                    "disk_write_mb": "50.0",
                    "io_intensity": "10.0",
                    "disk_type": "SSD",
                    "disk_speed_class": "2",
                    "runtime_sec": f"{10.0 + i * 0.1:.2f}",
                }
            )


# ---------------------------------------------------------------------------
# cetp --version
# ---------------------------------------------------------------------------


def test_version():
    runner = CliRunner()
    result = runner.invoke(main, ["--version"])
    assert "0.1.0" in result.output


# ---------------------------------------------------------------------------
# cetp profile
# ---------------------------------------------------------------------------


def test_profile_runs_successfully():
    runner = CliRunner()
    result = runner.invoke(main, ["profile"])
    assert result.exit_code == 0


def test_profile_shows_cpu_cores():
    runner = CliRunner()
    result = runner.invoke(main, ["profile"])
    assert "CPU cores" in result.output


def test_profile_shows_ram():
    runner = CliRunner()
    result = runner.invoke(main, ["profile"])
    assert "Total RAM" in result.output


def test_profile_shows_disk_type():
    runner = CliRunner()
    result = runner.invoke(main, ["profile"])
    assert "Disk type" in result.output


def test_profile_json_flag():
    runner = CliRunner()
    result = runner.invoke(main, ["profile", "--json"])
    assert result.exit_code == 0
    data = json.loads(result.output)
    assert "cpu_cores" in data


def test_profile_json_has_all_keys():
    runner = CliRunner()
    result = runner.invoke(main, ["profile", "--json"])
    assert result.exit_code == 0
    data = json.loads(result.output)
    for key in (
        "cpu_cores",
        "memory_total_gb",
        "disk_type",
        "disk_speed_class",
        "platform",
        "hostname",
    ):
        assert key in data, f"Key '{key}' missing from JSON output"


# ---------------------------------------------------------------------------
# cetp schema
# ---------------------------------------------------------------------------


def test_schema_shows_required_columns():
    runner = CliRunner()
    result = runner.invoke(main, ["schema"])
    assert "Required columns" in result.output


def test_schema_shows_constraints():
    runner = CliRunner()
    result = runner.invoke(main, ["schema"])
    assert "Constraints" in result.output


def test_schema_shows_minimum_rows():
    runner = CliRunner()
    result = runner.invoke(main, ["schema"])
    assert str(MIN_ROW_COUNT) in result.output


def test_schema_shows_valid_models():
    """workload_type (ML/DB/WEB) no longer exists in the schema — the
    constrained categorical column is now 'model', with values resnet18/
    resnet50/mobilenet/distilbert."""
    runner = CliRunner()
    result = runner.invoke(main, ["schema"])
    for model_name in sorted(VALID_MODELS):
        assert model_name in result.output


def test_schema_export_creates_file(tmp_path):
    export_file = str(tmp_path / "test_template.csv")
    runner = CliRunner()
    result = runner.invoke(main, ["schema", "--export", export_file])
    assert result.exit_code == 0
    assert os.path.exists(export_file)


def test_schema_export_confirmation_message(tmp_path):
    export_file = str(tmp_path / "test_template.csv")
    runner = CliRunner()
    result = runner.invoke(main, ["schema", "--export", export_file])
    assert "Template exported" in result.output


def test_schema_export_file_has_headers(tmp_path):
    export_file = str(tmp_path / "test_template.csv")
    runner = CliRunner()
    runner.invoke(main, ["schema", "--export", export_file])
    with open(export_file, newline="", encoding="utf-8") as fh:
        reader = csv.reader(fh)
        headers = next(reader)
    for col in REQUIRED_COLUMNS:
        assert col in headers, f"Column '{col}' missing from exported template"


# ---------------------------------------------------------------------------
# cetp validate
# ---------------------------------------------------------------------------


def test_validate_missing_data_flag():
    runner = CliRunner()
    result = runner.invoke(main, ["validate"])
    assert result.exit_code != 0
    assert "data" in result.output.lower() or "missing" in result.output.lower()


def test_validate_nonexistent_file():
    runner = CliRunner()
    result = runner.invoke(main, ["validate", "--data", "/nonexistent/path.csv"])
    assert result.exit_code == 1


def test_validate_nonexistent_file_message():
    runner = CliRunner()
    result = runner.invoke(main, ["validate", "--data", "/nonexistent/path.csv"])
    assert "not found" in result.output.lower() or "/nonexistent" in result.output


def test_validate_valid_csv(tmp_path):
    csv_file = str(tmp_path / "valid.csv")
    _make_valid_csv(csv_file, n_rows=150)
    runner = CliRunner()
    result = runner.invoke(main, ["validate", "--data", csv_file])
    assert result.exit_code == 0, result.output


def test_validate_valid_csv_shows_row_count(tmp_path):
    csv_file = str(tmp_path / "valid.csv")
    _make_valid_csv(csv_file, n_rows=150)
    runner = CliRunner()
    result = runner.invoke(main, ["validate", "--data", csv_file])
    assert "150" in result.output


def test_validate_invalid_csv_exits_nonzero(tmp_path):
    csv_file = str(tmp_path / "small.csv")
    _make_valid_csv(csv_file, n_rows=10)
    runner = CliRunner()
    result = runner.invoke(main, ["validate", "--data", csv_file])
    assert result.exit_code == 1


def test_validate_shows_violations(tmp_path):
    csv_file = str(tmp_path / "small.csv")
    _make_valid_csv(csv_file, n_rows=10)
    runner = CliRunner()
    result = runner.invoke(main, ["validate", "--data", csv_file])
    assert result.exit_code == 1
    assert f"Insufficient data: 10 rows found, minimum required is {MIN_ROW_COUNT}" in result.output


# ---------------------------------------------------------------------------
# cetp info
# ---------------------------------------------------------------------------


def test_info_runs_successfully():
    runner = CliRunner()
    result = runner.invoke(main, ["info"])
    assert result.exit_code == 0


def test_info_shows_version():
    runner = CliRunner()
    result = runner.invoke(main, ["info"])
    assert "0.1.0" in result.output


def test_info_shows_model_status():
    runner = CliRunner()
    result = runner.invoke(main, ["info"])
    assert "Model status" in result.output


def test_info_shows_sla_thresholds():
    runner = CliRunner()
    result = runner.invoke(main, ["info"])
    assert "SLA Thresholds" in result.output


def test_info_shows_all_four_model_thresholds():
    """SLA thresholds are now keyed by ML model (resnet18/resnet50/mobilenet/
    distilbert, see sla_defaults.json), not by the old workload_type
    (ML/DB/WEB) categories, which no longer exist anywhere in the schema."""
    runner = CliRunner()
    result = runner.invoke(main, ["info"])
    for model_name in sorted(VALID_MODELS):
        assert model_name in result.output


def test_info_shows_resnet18_warn_and_sla_values():
    """Regression test for the info command's SLA loop: it used to iterate
    over hardcoded ('ML', 'DB', 'WEB') keys against a model-keyed sla dict,
    so it silently printed the section header with zero threshold lines."""
    runner = CliRunner()
    result = runner.invoke(main, ["info"])
    assert "resnet18" in result.output
    assert "warn at 234.7s" in result.output
    assert "SLA limit 393.8s" in result.output


def test_info_shows_active_model_status():
    """The repo ships a real base_model.pkl artefact (model/artifacts/), so
    `cetp info` must report it as active rather than the 'no model' fallback."""
    runner = CliRunner()
    result = runner.invoke(main, ["info"])
    assert "Base model active" in result.output


# ---------------------------------------------------------------------------
# cetp predict
# ---------------------------------------------------------------------------


def test_predict_requires_model():
    runner = CliRunner()
    result = runner.invoke(main, ["predict", "--complexity", "3"])
    assert result.exit_code != 0


def test_predict_requires_complexity():
    runner = CliRunner()
    result = runner.invoke(main, ["predict", "--model", "resnet18"])
    assert result.exit_code != 0


def test_predict_rejects_invalid_model_choice():
    """--model only accepts the 4 known models; anything else is a click UsageError."""
    runner = CliRunner()
    result = runner.invoke(main, ["predict", "--model", "bert-large", "--complexity", "3"])
    assert result.exit_code != 0


def test_predict_rejects_complexity_out_of_range():
    runner = CliRunner()
    result = runner.invoke(main, ["predict", "--model", "resnet18", "--complexity", "6"])
    assert result.exit_code != 0


def test_predict_success_prints_result():
    """A valid in-range prediction against the real base model succeeds and
    prints the documented human-readable sections."""
    runner = CliRunner()
    result = runner.invoke(
        main,
        [
            "predict",
            "--model",
            "resnet18",
            "--complexity",
            "3",
            "--cpu-cores",
            "4",
            "--memory-gb",
            "8",
        ],
    )
    assert result.exit_code in (0, 1), result.output  # 1 only via --fail-on-red, not set here
    assert "=== Prediction Result ===" in result.output
    assert "Model: resnet18 (complexity 3)" in result.output
    assert "Predicted runtime:" in result.output
    assert "Confidence interval:" in result.output
    assert "Confidence status: RELIABLE" in result.output
    assert "SLA status:" in result.output
    assert "Top factors:" in result.output


def test_predict_extrapolated_hardware_shows_warning_block():
    """Hardware far outside the trained range must surface a visible warning
    block naming every violated range, before the rest of the output."""
    runner = CliRunner()
    result = runner.invoke(
        main,
        [
            "predict",
            "--model",
            "resnet18",
            "--complexity",
            "3",
            "--cpu-cores",
            "32",
            "--memory-gb",
            "64",
        ],
    )
    assert result.exit_code in (0, 1), result.output
    assert "WARNING" in result.output
    assert "cpu_count=32 is outside the trained range" in result.output
    assert "total_memory_mb=65536.0" in result.output
    assert "Confidence status: EXTRAPOLATED" in result.output
    # The warning block must appear before the prediction result section.
    assert result.output.index("WARNING") < result.output.index("=== Prediction Result ===")


def test_predict_memory_gb_converts_using_binary_1024(monkeypatch):
    """--memory-gb 8 must become total_memory_mb=8192.0 (binary GB, matching
    psutil.virtual_memory().total-derived training data), not 8000.0 (decimal).
    total_memory_mb isn't echoed back in predictor.predict()'s result dict, so
    the actual value passed in is captured directly by spying on the method."""
    from cetp.predictor import CETPPredictor

    captured = {}
    original_predict = CETPPredictor.predict

    def _spy_predict(self, model_name, complexity_level, cpu_count, total_memory_mb):
        captured["total_memory_mb"] = total_memory_mb
        return original_predict(self, model_name, complexity_level, cpu_count, total_memory_mb)

    monkeypatch.setattr(CETPPredictor, "predict", _spy_predict)

    runner = CliRunner()
    result = runner.invoke(
        main,
        [
            "predict",
            "--model",
            "resnet18",
            "--complexity",
            "3",
            "--cpu-cores",
            "4",
            "--memory-gb",
            "8",
        ],
    )
    assert result.exit_code in (0, 1), result.output
    assert captured["total_memory_mb"] == 8192.0


def test_predict_json_output_is_valid_json():
    runner = CliRunner()
    result = runner.invoke(
        main,
        [
            "predict",
            "--model",
            "resnet18",
            "--complexity",
            "3",
            "--cpu-cores",
            "4",
            "--memory-gb",
            "8",
            "--json",
        ],
    )
    assert result.exit_code in (0, 1), result.output
    payload = json.loads(result.output)
    assert "predicted_runtime_sec" in payload
    assert "sla_flag" in payload
    assert "shap_features" in payload


def test_predict_fail_on_red_exits_nonzero_when_red():
    """resnet18's SLA limit is now grounded in the real training distribution's
    p90 (393.8s, see sla_defaults.json) — complexity 4 on this hardware predicts
    an upper bound (~416s) above that limit, so --fail-on-red must force exit 1."""
    runner = CliRunner()
    result = runner.invoke(
        main,
        [
            "predict",
            "--model",
            "resnet18",
            "--complexity",
            "4",
            "--cpu-cores",
            "4",
            "--memory-gb",
            "8",
            "--fail-on-red",
        ],
    )
    assert "SLA status: RED" in result.output
    assert result.exit_code == 1


# ---------------------------------------------------------------------------
# cetp train
# ---------------------------------------------------------------------------


def test_train_without_data_flag():
    runner = CliRunner()
    result = runner.invoke(main, ["train"])
    assert result.exit_code != 0


def test_train_nonexistent_file():
    """--data uses click.Path(exists=True), so a missing file is rejected by
    Click's own option parsing (exit code 2, a UsageError) before the train
    command body — and thus trainer.train() — ever runs."""
    runner = CliRunner()
    result = runner.invoke(main, ["train", "--data", "/nonexistent.csv"])
    assert result.exit_code == 2
    assert "does not exist" in result.output


def test_train_invalid_csv_shows_violations(tmp_path):
    csv_file = str(tmp_path / "small.csv")
    _make_valid_csv(csv_file, n_rows=10)
    runner = CliRunner()
    result = runner.invoke(main, ["train", "--data", csv_file])
    assert result.exit_code == 1
    assert f"Insufficient data: 10 rows found, minimum required is {MIN_ROW_COUNT}" in result.output


def test_train_schema_violation_csv_shows_violations(tmp_path):
    """500-row CSV with invalid workload_type → ValidationError (not InsufficientDataError)."""
    csv_file = str(tmp_path / "bad_schema.csv")
    _make_schema_violation_csv(csv_file, n_rows=500)
    runner = CliRunner()
    result = runner.invoke(main, ["train", "--data", csv_file])
    assert result.exit_code == 1
    assert "INVALID" in result.output or "workload_type" in result.output or "✗" in result.output


def test_train_invokes_trainer_with_correct_call_pattern(tmp_path):
    """Regression test for the CETPTrainer(output_dir=...) / train(csv_path, sla_path)
    call pattern: output_dir is an __init__ argument, not a train() argument, and
    train() takes exactly (csv_path, sla_path). Passing output_dir into train() as a
    third positional/keyword argument would raise TypeError — this exercises the
    real cetp train CLI command end-to-end against a schema-correct CSV to confirm
    that regression can't reappear silently."""
    from tests.test_trainer import make_training_csv

    csv_path = make_training_csv(tmp_path / "train.csv")
    output_dir = tmp_path / "cetp_out"
    sla_path = tmp_path / "sla.json"
    sla_path.write_text(json.dumps({"resnet18": {"sla_runtime_sec": 300.0, "warn_at_sec": 200.0}}))

    runner = CliRunner()
    result = runner.invoke(
        main,
        [
            "train",
            "--data",
            csv_path,
            "--sla",
            str(sla_path),
            "--output-dir",
            str(output_dir),
        ],
    )

    assert result.exit_code == 0, result.output
    assert "Model saved to" in result.output
    assert (output_dir / "custom_model.pkl").exists()
    assert (output_dir / "custom_training_range.json").exists()
    assert (output_dir / "custom_model_meta.json").exists()


# ---------------------------------------------------------------------------
# cetp validate — ValidationError branch (schema violations, not row count)
# ---------------------------------------------------------------------------


def test_validate_schema_violation_csv(tmp_path):
    """500-row CSV with invalid workload_type triggers ValidationError, not InsufficientDataError."""
    csv_file = str(tmp_path / "bad_schema.csv")
    _make_schema_violation_csv(csv_file, n_rows=500)
    runner = CliRunner()
    result = runner.invoke(main, ["validate", "--data", csv_file])
    assert result.exit_code == 1
    assert "INVALID" in result.output or "workload_type" in result.output or "✗" in result.output


def test_validate_schema_violation_shows_violations(tmp_path):
    csv_file = str(tmp_path / "bad_schema.csv")
    _make_schema_violation_csv(csv_file, n_rows=500)
    runner = CliRunner()
    result = runner.invoke(main, ["validate", "--data", csv_file])
    assert "✗" in result.output
