"""Unit tests for cetp.validator — Phase 4 Schema Validator."""

import csv
from typing import Optional

import pytest

from cetp.validator import (
    MIN_ROW_COUNT,
    NUMERIC_COLUMNS,
    REQUIRED_COLUMNS,
    VALID_MODELS,
    InsufficientDataError,
    ValidationError,
    export_schema_template,
    get_schema_info,
    validate_csv,
)


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------


def make_csv(tmp_path, rows: int = 120, overrides: Optional[dict] = None) -> str:
    """
    Creates a valid CSV file with `rows` rows at tmp_path/test.csv.
    overrides: dict mapping column name to a list of values (applied to rows
    in order; remaining rows use defaults) OR a single value applied to all rows.
    Returns the path as a string.
    """
    if overrides is None:
        overrides = {}

    models = ["resnet18", "resnet50", "mobilenet", "distilbert"]
    complexities = [1, 2, 3, 4, 5]
    cpu_count_list = [2, 3, 4]
    total_memory_list = [3911.8, 8192.0, 15785.3]
    batch_size_list = [8, 16, 32, 64, 128]
    num_iterations_list = [19, 250, 500, 973]

    runtime_sec = 14.37

    data_rows = []
    for i in range(rows):
        row = {
            "cpu_count": str(cpu_count_list[i % len(cpu_count_list)]),
            "total_memory_mb": str(total_memory_list[i % len(total_memory_list)]),
            "model": models[i % len(models)],
            "complexity_level": str(complexities[i % len(complexities)]),
            "batch_size": str(batch_size_list[i % len(batch_size_list)]),
            "num_iterations": str(num_iterations_list[i % len(num_iterations_list)]),
            "runtime_sec": str(runtime_sec),
        }
        data_rows.append(row)

    for col, value in overrides.items():
        if isinstance(value, list):
            for j, v in enumerate(value):
                if j < rows:
                    data_rows[j][col] = "" if v is None else str(v)
        else:
            for j in range(rows):
                data_rows[j][col] = "" if value is None else str(value)

    path = tmp_path / "test.csv"
    with open(path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=REQUIRED_COLUMNS)
        writer.writeheader()
        writer.writerows(data_rows)

    return str(path)


# ---------------------------------------------------------------------------
# File-level tests
# ---------------------------------------------------------------------------


class TestFileLevelBehavior:
    def test_file_not_found(self, tmp_path):
        """validate_csv raises FileNotFoundError for a nonexistent path."""
        with pytest.raises(FileNotFoundError):
            validate_csv(str(tmp_path / "nonexistent.csv"))

    def test_valid_csv_returns_dict(self, tmp_path):
        """A valid 120-row CSV returns a dict with valid=True and no violations."""
        path = make_csv(tmp_path)
        result = validate_csv(path)
        assert result["valid"] is True
        assert result["violations"] == []

    def test_valid_csv_row_count(self, tmp_path):
        """Returned dict contains the exact number of data rows."""
        path = make_csv(tmp_path, rows=150)
        result = validate_csv(path)
        assert result["row_count"] == 150

    def test_valid_csv_has_runtime_stats(self, tmp_path):
        """Returned dict includes runtime_stats with min, max, mean keys."""
        path = make_csv(tmp_path)
        result = validate_csv(path)
        assert "runtime_stats" in result
        stats = result["runtime_stats"]
        assert "min" in stats
        assert "max" in stats
        assert "mean" in stats

    def test_valid_csv_has_model_counts(self, tmp_path):
        """Returned dict includes model_counts covering all valid models."""
        path = make_csv(tmp_path)
        result = validate_csv(path)
        counts = result["model_counts"]
        for m in VALID_MODELS:
            assert m in counts


# ---------------------------------------------------------------------------
# Column validation tests
# ---------------------------------------------------------------------------


class TestColumnValidation:
    def test_missing_single_column(self, tmp_path):
        """CSV missing runtime_sec raises ValidationError mentioning runtime_sec."""
        cols = [c for c in REQUIRED_COLUMNS if c != "runtime_sec"]
        path = str(tmp_path / "missing.csv")
        with open(path, "w", newline="", encoding="utf-8") as fh:
            writer = csv.DictWriter(fh, fieldnames=cols)
            writer.writeheader()
            writer.writerows([{c: "1" for c in cols} for _ in range(120)])
        with pytest.raises(ValidationError) as exc_info:
            validate_csv(path)
        joined = " ".join(exc_info.value.violations)
        assert "runtime_sec" in joined

    def test_missing_multiple_columns(self, tmp_path):
        """CSV missing 3 required columns raises ValidationError listing all 3."""
        missing = ["runtime_sec", "cpu_count", "batch_size"]
        cols = [c for c in REQUIRED_COLUMNS if c not in missing]
        path = str(tmp_path / "missing3.csv")
        with open(path, "w", newline="", encoding="utf-8") as fh:
            writer = csv.DictWriter(fh, fieldnames=cols)
            writer.writeheader()
            writer.writerows([{c: "1" for c in cols} for _ in range(120)])
        with pytest.raises(ValidationError) as exc_info:
            validate_csv(path)
        joined = " ".join(exc_info.value.violations)
        for col in missing:
            assert col in joined

    def test_extra_columns_are_ignored(self, tmp_path):
        """A CSV with extra columns beyond REQUIRED_COLUMNS still passes validation."""
        path = make_csv(tmp_path)
        extra_path = str(tmp_path / "extra.csv")
        with open(path, newline="", encoding="utf-8") as fh:
            reader = csv.DictReader(fh)
            original_rows = list(reader)
        fieldnames = REQUIRED_COLUMNS + ["extra_col"]
        for row in original_rows:
            row["extra_col"] = "ignored"
        with open(extra_path, "w", newline="", encoding="utf-8") as fh:
            writer = csv.DictWriter(fh, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(original_rows)
        result = validate_csv(extra_path)
        assert result["valid"] is True


# ---------------------------------------------------------------------------
# Row count tests
# ---------------------------------------------------------------------------


class TestRowCountValidation:
    def test_insufficient_rows_raises(self, tmp_path):
        """A CSV with fewer than MIN_ROW_COUNT rows raises InsufficientDataError."""
        path = make_csv(tmp_path, rows=MIN_ROW_COUNT - 1)
        with pytest.raises(InsufficientDataError):
            validate_csv(path)

    def test_exactly_100_rows_passes(self, tmp_path):
        """Exactly MIN_ROW_COUNT rows is the minimum accepted."""
        path = make_csv(tmp_path, rows=MIN_ROW_COUNT)
        result = validate_csv(path)
        assert result["valid"] is True

    def test_insufficient_error_is_validation_error(self, tmp_path):
        """InsufficientDataError is a subclass of ValidationError."""
        path = make_csv(tmp_path, rows=10)
        with pytest.raises(ValidationError):
            validate_csv(path)
        assert issubclass(InsufficientDataError, ValidationError)


# ---------------------------------------------------------------------------
# runtime_sec tests
# ---------------------------------------------------------------------------


class TestRuntimeSec:
    def test_negative_runtime_raises(self, tmp_path):
        """A row with runtime_sec=-1.0 raises ValidationError."""
        path = make_csv(tmp_path, overrides={"runtime_sec": ["-1.0"]})
        with pytest.raises(ValidationError) as exc_info:
            validate_csv(path)
        assert any("runtime_sec" in v for v in exc_info.value.violations)

    def test_zero_runtime_raises(self, tmp_path):
        """A row with runtime_sec=0.0 raises ValidationError (must be strictly > 0)."""
        path = make_csv(tmp_path, overrides={"runtime_sec": ["0.0"]})
        with pytest.raises(ValidationError) as exc_info:
            validate_csv(path)
        assert any("runtime_sec" in v for v in exc_info.value.violations)

    def test_valid_runtime_passes(self, tmp_path):
        """runtime_sec=0.001 is strictly > 0 and must pass validation."""
        path = make_csv(tmp_path, overrides={"runtime_sec": "0.001"})
        result = validate_csv(path)
        assert result["valid"] is True


# ---------------------------------------------------------------------------
# total_memory_mb tests
# ---------------------------------------------------------------------------


class TestTotalMemoryMb:
    def test_negative_total_memory_raises(self, tmp_path):
        """total_memory_mb=-100.0 is not positive and raises ValidationError."""
        path = make_csv(tmp_path, overrides={"total_memory_mb": ["-100.0"]})
        with pytest.raises(ValidationError) as exc_info:
            validate_csv(path)
        assert any("total_memory_mb" in v for v in exc_info.value.violations)

    def test_zero_total_memory_raises(self, tmp_path):
        """total_memory_mb=0.0 is not positive and raises ValidationError."""
        path = make_csv(tmp_path, overrides={"total_memory_mb": ["0.0"]})
        with pytest.raises(ValidationError) as exc_info:
            validate_csv(path)
        assert any("total_memory_mb" in v for v in exc_info.value.violations)

    def test_valid_total_memory_passes(self, tmp_path):
        """total_memory_mb=15785.3 is a positive float and must pass."""
        path = make_csv(tmp_path, overrides={"total_memory_mb": "15785.3"})
        result = validate_csv(path)
        assert result["valid"] is True


# ---------------------------------------------------------------------------
# model tests
# ---------------------------------------------------------------------------


class TestModel:
    def test_invalid_model_raises(self, tmp_path):
        """model='INVALID' is not in VALID_MODELS and raises ValidationError."""
        assert "INVALID" not in VALID_MODELS
        path = make_csv(tmp_path, overrides={"model": ["INVALID"]})
        with pytest.raises(ValidationError) as exc_info:
            validate_csv(path)
        assert any("model" in v for v in exc_info.value.violations)

    def test_valid_models_pass(self, tmp_path):
        """All values in VALID_MODELS pass validation."""
        path = make_csv(tmp_path)
        result = validate_csv(path)
        assert result["valid"] is True
        assert set(result["model_counts"].keys()) == VALID_MODELS


# ---------------------------------------------------------------------------
# complexity_level tests
# ---------------------------------------------------------------------------


class TestComplexityLevel:
    def test_complexity_out_of_range_raises(self, tmp_path):
        """complexity_level=6 exceeds [1, 5] and raises ValidationError."""
        path = make_csv(tmp_path, overrides={"complexity_level": ["6"]})
        with pytest.raises(ValidationError) as exc_info:
            validate_csv(path)
        assert any("complexity_level" in v for v in exc_info.value.violations)

    def test_complexity_zero_raises(self, tmp_path):
        """complexity_level=0 is below [1, 5] and raises ValidationError."""
        path = make_csv(tmp_path, overrides={"complexity_level": ["0"]})
        with pytest.raises(ValidationError) as exc_info:
            validate_csv(path)
        assert any("complexity_level" in v for v in exc_info.value.violations)


# ---------------------------------------------------------------------------
# Empty value tests
# ---------------------------------------------------------------------------


class TestEmptyValues:
    def test_empty_numeric_field_raises(self, tmp_path):
        """A row with an empty string in any NUMERIC_COLUMNS field raises ValidationError."""
        assert "cpu_count" in NUMERIC_COLUMNS
        path = make_csv(tmp_path, overrides={"cpu_count": [""]})
        with pytest.raises(ValidationError) as exc_info:
            validate_csv(path)
        assert any("cpu_count" in v for v in exc_info.value.violations)


# ---------------------------------------------------------------------------
# Violation collection tests
# ---------------------------------------------------------------------------


class TestViolationCollection:
    def test_multiple_violations_collected(self, tmp_path):
        """All violations from multiple bad rows are collected, not just the first."""
        path = make_csv(
            tmp_path,
            overrides={"model": ["BAD_1", "BAD_2", "BAD_3"]},
        )
        with pytest.raises(ValidationError) as exc_info:
            validate_csv(path)
        assert len(exc_info.value.violations) >= 3

    def test_violation_includes_row_number(self, tmp_path):
        """Each violation message includes a row number for easy data lookup."""
        path = make_csv(tmp_path, overrides={"runtime_sec": ["-5.0"]})
        with pytest.raises(ValidationError) as exc_info:
            validate_csv(path)
        assert any("Row" in v for v in exc_info.value.violations)
        assert any("2" in v for v in exc_info.value.violations)


# ---------------------------------------------------------------------------
# Additional constraint tests (cpu_count, batch_size, num_iterations, non-numeric values)
# ---------------------------------------------------------------------------


class TestAdditionalConstraints:
    def test_cpu_count_zero_raises(self, tmp_path):
        """cpu_count=0 is not a positive integer and raises ValidationError."""
        path = make_csv(tmp_path, overrides={"cpu_count": ["0"]})
        with pytest.raises(ValidationError) as exc_info:
            validate_csv(path)
        assert any("cpu_count" in v for v in exc_info.value.violations)

    def test_cpu_count_negative_raises(self, tmp_path):
        """cpu_count=-1 is not a positive integer and raises ValidationError."""
        path = make_csv(tmp_path, overrides={"cpu_count": ["-1"]})
        with pytest.raises(ValidationError) as exc_info:
            validate_csv(path)
        assert any("cpu_count" in v for v in exc_info.value.violations)

    def test_batch_size_invalid_raises(self, tmp_path):
        """batch_size=0 is not a positive integer and raises ValidationError."""
        path = make_csv(tmp_path, overrides={"batch_size": ["0"]})
        with pytest.raises(ValidationError) as exc_info:
            validate_csv(path)
        assert any("batch_size" in v for v in exc_info.value.violations)

    def test_num_iterations_invalid_raises(self, tmp_path):
        """num_iterations=-1 is not a positive integer and raises ValidationError."""
        path = make_csv(tmp_path, overrides={"num_iterations": ["-1"]})
        with pytest.raises(ValidationError) as exc_info:
            validate_csv(path)
        assert any("num_iterations" in v for v in exc_info.value.violations)

    def test_non_numeric_value_in_numeric_column_raises(self, tmp_path):
        """A non-numeric string in a NUMERIC_COLUMNS field raises ValidationError."""
        path = make_csv(tmp_path, overrides={"cpu_count": ["not_a_number"]})
        with pytest.raises(ValidationError) as exc_info:
            validate_csv(path)
        assert any("cpu_count" in v for v in exc_info.value.violations)

    def test_violation_limit_caps_at_50(self, tmp_path):
        """When 50+ rows are bad, collection stops at 50 violations to save memory."""
        path = make_csv(tmp_path, overrides={"model": ["BAD"] * 55}, rows=120)
        with pytest.raises(ValidationError) as exc_info:
            validate_csv(path)
        assert len(exc_info.value.violations) == 50


# ---------------------------------------------------------------------------
# export_schema_template tests
# ---------------------------------------------------------------------------


class TestExportSchemaTemplate:
    def test_export_creates_file(self, tmp_path):
        """export_schema_template creates a file at the given path."""
        out = tmp_path / "template.csv"
        export_schema_template(str(out))
        assert out.exists()

    def test_export_has_correct_headers(self, tmp_path):
        """Exported template contains all REQUIRED_COLUMNS as headers."""
        out = tmp_path / "template.csv"
        export_schema_template(str(out))
        with open(out, newline="", encoding="utf-8") as fh:
            reader = csv.DictReader(fh)
            headers = reader.fieldnames or []
        for col in REQUIRED_COLUMNS:
            assert col in headers, f"Column '{col}' missing from template"

    def test_export_has_example_row(self, tmp_path):
        """Exported template has exactly one example data row (header + 1 row total)."""
        out = tmp_path / "template.csv"
        export_schema_template(str(out))
        with open(out, newline="", encoding="utf-8") as fh:
            reader = csv.DictReader(fh)
            data_rows = list(reader)
        assert len(data_rows) == 1

    def test_export_example_values(self, tmp_path):
        """Example row contains model=resnet18 and runtime_sec=14.37."""
        out = tmp_path / "template.csv"
        export_schema_template(str(out))
        with open(out, newline="", encoding="utf-8") as fh:
            reader = csv.DictReader(fh)
            row = list(reader)[0]
        assert row["model"] == "resnet18"
        assert row["runtime_sec"] == "14.37"


# ---------------------------------------------------------------------------
# get_schema_info tests
# ---------------------------------------------------------------------------


class TestGetSchemaInfo:
    def test_schema_info_keys(self):
        """get_schema_info returns a dict with all expected top-level keys."""
        info = get_schema_info()
        expected = {
            "required_columns",
            "numeric_columns",
            "valid_models",
            "min_row_count",
            "column_constraints",
        }
        assert set(info.keys()) == expected

    def test_schema_info_min_row_count(self):
        """get_schema_info reports min_row_count of 100."""
        info = get_schema_info()
        assert info["min_row_count"] == MIN_ROW_COUNT
        assert info["min_row_count"] == 100
