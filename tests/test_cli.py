"""Tests for cetp CLI — Phase 6."""

import csv
import json
import os

from click.testing import CliRunner

from cetp.cli import main


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_valid_csv(path: str, n_rows: int = 500) -> None:
    """Write a CETP-schema-valid CSV with n_rows data rows to path."""
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
    types = ["ML", "DB", "WEB"]
    with open(path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        for i in range(n_rows):
            wt = types[i % 3]
            writer.writerow(
                {
                    "run_id": str(i),
                    "workload_type": wt,
                    "workload_name": f"wl_{i}",
                    "workload_complexity": str((i % 5) + 1),
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
    assert "500" in result.output


def test_schema_shows_workload_types():
    runner = CliRunner()
    result = runner.invoke(main, ["schema"])
    assert "ML" in result.output
    assert "DB" in result.output
    assert "WEB" in result.output


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
    required = [
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
    for col in required:
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
    _make_valid_csv(csv_file, n_rows=500)
    runner = CliRunner()
    result = runner.invoke(main, ["validate", "--data", csv_file])
    assert result.exit_code == 0


def test_validate_valid_csv_shows_row_count(tmp_path):
    csv_file = str(tmp_path / "valid.csv")
    _make_valid_csv(csv_file, n_rows=500)
    runner = CliRunner()
    result = runner.invoke(main, ["validate", "--data", csv_file])
    assert "500" in result.output


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
    # InsufficientDataError message contains row count and minimum
    assert "10" in result.output or "Insufficient" in result.output or "500" in result.output


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


def test_info_shows_ml_threshold():
    runner = CliRunner()
    result = runner.invoke(main, ["info"])
    assert "ML" in result.output


def test_info_shows_db_threshold():
    runner = CliRunner()
    result = runner.invoke(main, ["info"])
    assert "DB" in result.output


def test_info_shows_web_threshold():
    runner = CliRunner()
    result = runner.invoke(main, ["info"])
    assert "WEB" in result.output


def test_info_no_model_message():
    """No model is trained in the CI/dev environment, so this message is expected."""
    runner = CliRunner()
    result = runner.invoke(main, ["info"])
    assert "No model artefact found" in result.output


# ---------------------------------------------------------------------------
# cetp predict
# ---------------------------------------------------------------------------


def test_predict_without_model_exits_nonzero():
    runner = CliRunner()
    result = runner.invoke(
        main,
        [
            "predict",
            "--workload-type",
            "ML",
            "--workload-name",
            "test_wl",
            "--complexity",
            "3",
        ],
    )
    assert result.exit_code == 1


def test_predict_without_model_shows_message():
    runner = CliRunner()
    result = runner.invoke(
        main,
        [
            "predict",
            "--workload-type",
            "ML",
            "--workload-name",
            "test_wl",
            "--complexity",
            "3",
        ],
    )
    assert "model" in result.output.lower() or "artefact" in result.output.lower()


def test_predict_requires_workload_type():
    runner = CliRunner()
    result = runner.invoke(
        main,
        [
            "predict",
            "--workload-name",
            "test_wl",
            "--complexity",
            "3",
        ],
    )
    assert result.exit_code != 0


def test_predict_requires_workload_name():
    runner = CliRunner()
    result = runner.invoke(
        main,
        [
            "predict",
            "--workload-type",
            "ML",
            "--complexity",
            "3",
        ],
    )
    assert result.exit_code != 0


def test_predict_requires_complexity():
    runner = CliRunner()
    result = runner.invoke(
        main,
        [
            "predict",
            "--workload-type",
            "ML",
            "--workload-name",
            "test_wl",
        ],
    )
    assert result.exit_code != 0


# ---------------------------------------------------------------------------
# cetp train
# ---------------------------------------------------------------------------


def test_train_without_data_flag():
    runner = CliRunner()
    result = runner.invoke(main, ["train"])
    assert result.exit_code != 0


def test_train_nonexistent_file():
    runner = CliRunner()
    result = runner.invoke(main, ["train", "--data", "/nonexistent.csv"])
    assert result.exit_code == 1


def test_train_invalid_csv_shows_violations(tmp_path):
    csv_file = str(tmp_path / "small.csv")
    _make_valid_csv(csv_file, n_rows=10)
    runner = CliRunner()
    result = runner.invoke(main, ["train", "--data", csv_file])
    assert result.exit_code == 1
    assert "10" in result.output or "Insufficient" in result.output or "500" in result.output


def test_train_schema_violation_csv_shows_violations(tmp_path):
    """500-row CSV with invalid workload_type → ValidationError (not InsufficientDataError)."""
    csv_file = str(tmp_path / "bad_schema.csv")
    _make_schema_violation_csv(csv_file, n_rows=500)
    runner = CliRunner()
    result = runner.invoke(main, ["train", "--data", csv_file])
    assert result.exit_code == 1
    assert "INVALID" in result.output or "workload_type" in result.output or "✗" in result.output


def test_train_valid_csv_missing_hyperparams(tmp_path):
    """Valid 500-row CSV passes validation and reaches trainer, which raises FileNotFoundError."""
    csv_file = str(tmp_path / "valid.csv")
    _make_valid_csv(csv_file, n_rows=500)
    runner = CliRunner()
    result = runner.invoke(main, ["train", "--data", csv_file])
    assert result.exit_code == 1
    assert "hyperparameters" in result.output.lower() or "base_hyperparams" in result.output


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
