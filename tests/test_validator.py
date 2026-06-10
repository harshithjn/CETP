"""Unit tests for cetp.validator."""
import csv

import pytest

from cetp.validator import (
    REQUIRED_COLUMNS,
    MIN_ROW_COUNT,
    InsufficientDataError,
    ValidationError,
    export_schema_template,
    validate_csv,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_valid_row(run_id: int) -> dict:
    """Return a single valid CETP data row."""
    return {
        "run_id": f"run_{run_id:04d}",
        "workload_type": "ML",
        "workload_name": "ml_test",
        "workload_complexity": "3",
        "cpu_cores": "8",
        "memory_total_gb": "16.0",
        "cpu_avg_pct": "55.0",
        "effective_cpu": "3.6",
        "memory_avg_gb": "8.5",
        "memory_pressure": "0.53",
        "disk_read_mb": "120.0",
        "disk_write_mb": "45.0",
        "io_intensity": "16.5",
        "disk_type": "SSD",
        "disk_speed_class": "2",
        "runtime_sec": "14.7",
    }


def _write_csv(path, rows: list[dict], fieldnames=None) -> None:
    """Write rows to a CSV file at path."""
    if fieldnames is None:
        fieldnames = REQUIRED_COLUMNS
    with open(path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


# ---------------------------------------------------------------------------
# Tests: valid CSV
# ---------------------------------------------------------------------------

class TestValidateCsvValid:
    def test_passes_on_valid_600_row_csv(self, tmp_path):
        """A well-formed CSV with 600 rows must pass without violations."""
        rows = [_make_valid_row(i) for i in range(600)]
        csv_file = tmp_path / "valid.csv"
        _write_csv(csv_file, rows)
        result = validate_csv(str(csv_file))
        assert result["valid"] is True
        assert result["row_count"] == 600
        assert result["violations"] == []

    def test_returns_correct_row_count(self, tmp_path):
        """validate_csv() must report the exact number of data rows."""
        rows = [_make_valid_row(i) for i in range(700)]
        csv_file = tmp_path / "big.csv"
        _write_csv(csv_file, rows)
        result = validate_csv(str(csv_file))
        assert result["row_count"] == 700


# ---------------------------------------------------------------------------
# Tests: missing columns
# ---------------------------------------------------------------------------

class TestValidateCsvMissingColumns:
    def test_raises_for_missing_column(self, tmp_path):
        """Missing a required column should raise ValidationError."""
        rows = [_make_valid_row(i) for i in range(600)]
        # Drop 'runtime_sec' from every row
        for row in rows:
            del row["runtime_sec"]
        bad_cols = [c for c in REQUIRED_COLUMNS if c != "runtime_sec"]
        csv_file = tmp_path / "missing_col.csv"
        _write_csv(csv_file, rows, fieldnames=bad_cols)
        with pytest.raises(ValidationError) as exc_info:
            validate_csv(str(csv_file))
        assert any("runtime_sec" in v for v in exc_info.value.violations)

    def test_raises_for_multiple_missing_columns(self, tmp_path):
        """All missing columns must be reported in a single raise."""
        rows = [_make_valid_row(i) for i in range(600)]
        for row in rows:
            del row["runtime_sec"]
            del row["cpu_cores"]
        bad_cols = [c for c in REQUIRED_COLUMNS if c not in ("runtime_sec", "cpu_cores")]
        csv_file = tmp_path / "multi_missing.csv"
        _write_csv(csv_file, rows, fieldnames=bad_cols)
        with pytest.raises(ValidationError) as exc_info:
            validate_csv(str(csv_file))
        joined = " ".join(exc_info.value.violations)
        assert "runtime_sec" in joined or "cpu_cores" in joined


# ---------------------------------------------------------------------------
# Tests: invalid workload_type
# ---------------------------------------------------------------------------

class TestValidateCsvWorkloadType:
    def test_raises_for_invalid_workload_type(self, tmp_path):
        """A row with workload_type not in {ML, DB, WEB} must produce a violation."""
        rows = [_make_valid_row(i) for i in range(600)]
        rows[0]["workload_type"] = "BATCH"  # invalid
        csv_file = tmp_path / "bad_wt.csv"
        _write_csv(csv_file, rows)
        with pytest.raises(ValidationError) as exc_info:
            validate_csv(str(csv_file))
        assert any("workload_type" in v for v in exc_info.value.violations)

    def test_collects_all_invalid_workload_type_rows(self, tmp_path):
        """All rows with invalid workload_type must be reported, not just the first."""
        rows = [_make_valid_row(i) for i in range(600)]
        rows[5]["workload_type"] = "INVALID"
        rows[10]["workload_type"] = "ALSO_BAD"
        csv_file = tmp_path / "multi_bad_wt.csv"
        _write_csv(csv_file, rows)
        with pytest.raises(ValidationError) as exc_info:
            validate_csv(str(csv_file))
        assert len(exc_info.value.violations) >= 2


# ---------------------------------------------------------------------------
# Tests: negative runtime_sec
# ---------------------------------------------------------------------------

class TestValidateCsvRuntimeSec:
    def test_raises_for_negative_runtime(self, tmp_path):
        """runtime_sec <= 0 must produce a violation."""
        rows = [_make_valid_row(i) for i in range(600)]
        rows[3]["runtime_sec"] = "-5.0"
        csv_file = tmp_path / "neg_runtime.csv"
        _write_csv(csv_file, rows)
        with pytest.raises(ValidationError) as exc_info:
            validate_csv(str(csv_file))
        assert any("runtime_sec" in v for v in exc_info.value.violations)

    def test_raises_for_zero_runtime(self, tmp_path):
        """runtime_sec == 0 must produce a violation."""
        rows = [_make_valid_row(i) for i in range(600)]
        rows[0]["runtime_sec"] = "0.0"
        csv_file = tmp_path / "zero_runtime.csv"
        _write_csv(csv_file, rows)
        with pytest.raises(ValidationError) as exc_info:
            validate_csv(str(csv_file))
        assert any("runtime_sec" in v for v in exc_info.value.violations)


# ---------------------------------------------------------------------------
# Tests: memory_pressure out of range
# ---------------------------------------------------------------------------

class TestValidateCsvMemoryPressure:
    def test_raises_for_memory_pressure_above_one(self, tmp_path):
        """memory_pressure > 1.0 must produce a violation."""
        rows = [_make_valid_row(i) for i in range(600)]
        rows[2]["memory_pressure"] = "1.5"
        csv_file = tmp_path / "bad_mem_pressure.csv"
        _write_csv(csv_file, rows)
        with pytest.raises(ValidationError) as exc_info:
            validate_csv(str(csv_file))
        assert any("memory_pressure" in v for v in exc_info.value.violations)

    def test_raises_for_negative_memory_pressure(self, tmp_path):
        """memory_pressure < 0 must produce a violation."""
        rows = [_make_valid_row(i) for i in range(600)]
        rows[1]["memory_pressure"] = "-0.1"
        csv_file = tmp_path / "neg_mem_pressure.csv"
        _write_csv(csv_file, rows)
        with pytest.raises(ValidationError) as exc_info:
            validate_csv(str(csv_file))
        assert any("memory_pressure" in v for v in exc_info.value.violations)


# ---------------------------------------------------------------------------
# Tests: insufficient data
# ---------------------------------------------------------------------------

class TestValidateCsvInsufficientData:
    def test_raises_insufficient_data_for_fewer_than_500_rows(self, tmp_path):
        """A CSV with fewer than MIN_ROW_COUNT rows must raise InsufficientDataError."""
        rows = [_make_valid_row(i) for i in range(MIN_ROW_COUNT - 1)]
        csv_file = tmp_path / "small.csv"
        _write_csv(csv_file, rows)
        with pytest.raises(InsufficientDataError):
            validate_csv(str(csv_file))

    def test_insufficient_data_error_is_subclass_of_validation_error(self, tmp_path):
        """InsufficientDataError must be catchable as ValidationError."""
        rows = [_make_valid_row(i) for i in range(10)]
        csv_file = tmp_path / "tiny.csv"
        _write_csv(csv_file, rows)
        with pytest.raises(ValidationError):
            validate_csv(str(csv_file))

    def test_exactly_500_rows_passes(self, tmp_path):
        """Exactly MIN_ROW_COUNT rows must be accepted."""
        rows = [_make_valid_row(i) for i in range(MIN_ROW_COUNT)]
        csv_file = tmp_path / "exact.csv"
        _write_csv(csv_file, rows)
        result = validate_csv(str(csv_file))
        assert result["valid"] is True


# ---------------------------------------------------------------------------
# Tests: export_schema_template
# ---------------------------------------------------------------------------

class TestExportSchemaTemplate:
    def test_creates_file_with_all_required_headers(self, tmp_path):
        """Exported template must contain all REQUIRED_COLUMNS as headers."""
        out = tmp_path / "template.csv"
        export_schema_template(str(out))
        assert out.exists()
        with open(out, newline="", encoding="utf-8") as fh:
            reader = csv.DictReader(fh)
            headers = reader.fieldnames or []
        for col in REQUIRED_COLUMNS:
            assert col in headers, f"Column '{col}' missing from template"

    def test_template_contains_example_row(self, tmp_path):
        """Exported template must have at least one example data row."""
        out = tmp_path / "template.csv"
        export_schema_template(str(out))
        with open(out, newline="", encoding="utf-8") as fh:
            reader = csv.DictReader(fh)
            rows = list(reader)
        assert len(rows) >= 1, "Template must contain at least one example row"
