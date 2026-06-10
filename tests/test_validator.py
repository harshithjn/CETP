"""Unit tests for cetp.validator — Phase 4 Schema Validator."""
import csv
from typing import Optional

import pytest

from cetp.validator import (
    MIN_ROW_COUNT,
    NUMERIC_COLUMNS,
    REQUIRED_COLUMNS,
    VALID_DISK_TYPES,
    VALID_WORKLOAD_TYPES,
    InsufficientDataError,
    ValidationError,
    export_schema_template,
    get_schema_info,
    validate_csv,
)


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------

def make_csv(tmp_path, rows: int = 600, overrides: Optional[dict] = None) -> str:
    """
    Creates a valid CSV file with `rows` rows at tmp_path/test.csv.
    overrides: dict mapping column name to a list of values (applied to rows
    in order; remaining rows use defaults) OR a single value applied to all rows.
    Returns the path as a string.
    """
    if overrides is None:
        overrides = {}

    workload_types = ["ML", "DB", "WEB"]
    workload_names = ["ml_resnet", "tpch_q3", "wrk_low"]
    complexities = [1, 2, 3, 4, 5]
    cpu_cores_list = [2, 4, 8, 16]
    memory_total_list = [4, 8, 16, 32]
    disk_types = ["HDD", "SSD", "NVMe"]
    disk_speed_classes = [1, 2, 3]

    cpu_avg_pct = 45.0
    memory_avg_gb = 4.0
    disk_read_mb = 120.5
    disk_write_mb = 34.2
    runtime_sec = 14.37

    data_rows = []
    for i in range(rows):
        cpu_cores = cpu_cores_list[i % len(cpu_cores_list)]
        memory_total_gb = memory_total_list[i % len(memory_total_list)]
        disk_type = disk_types[i % len(disk_types)]
        disk_speed_class = disk_speed_classes[i % len(disk_speed_classes)]

        effective_cpu = cpu_cores * (cpu_avg_pct / 100.0)
        memory_pressure = memory_avg_gb / memory_total_gb
        io_intensity = (disk_read_mb + disk_write_mb) / runtime_sec

        row = {
            "run_id": str(i),
            "workload_type": workload_types[i % len(workload_types)],
            "workload_name": workload_names[i % len(workload_names)],
            "workload_complexity": str(complexities[i % len(complexities)]),
            "cpu_cores": str(cpu_cores),
            "memory_total_gb": str(float(memory_total_gb)),
            "cpu_avg_pct": str(cpu_avg_pct),
            "effective_cpu": f"{effective_cpu:.3f}",
            "memory_avg_gb": str(memory_avg_gb),
            "memory_pressure": f"{memory_pressure:.4f}",
            "disk_read_mb": str(disk_read_mb),
            "disk_write_mb": str(disk_write_mb),
            "io_intensity": f"{io_intensity:.4f}",
            "disk_type": disk_type,
            "disk_speed_class": str(disk_speed_class),
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
        """A valid 600-row CSV returns a dict with valid=True and no violations."""
        path = make_csv(tmp_path)
        result = validate_csv(path)
        assert result["valid"] is True
        assert result["violations"] == []

    def test_valid_csv_row_count(self, tmp_path):
        """Returned dict contains the exact number of data rows."""
        path = make_csv(tmp_path, rows=700)
        result = validate_csv(path)
        assert result["row_count"] == 700

    def test_valid_csv_has_runtime_stats(self, tmp_path):
        """Returned dict includes runtime_stats with min, max, mean keys."""
        path = make_csv(tmp_path)
        result = validate_csv(path)
        assert "runtime_stats" in result
        stats = result["runtime_stats"]
        assert "min" in stats
        assert "max" in stats
        assert "mean" in stats

    def test_valid_csv_has_workload_counts(self, tmp_path):
        """Returned dict includes workload_type_counts covering all valid types."""
        path = make_csv(tmp_path)
        result = validate_csv(path)
        counts = result["workload_type_counts"]
        for wt in VALID_WORKLOAD_TYPES:
            assert wt in counts


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
            writer.writerows([{c: "1" for c in cols} for _ in range(600)])
        with pytest.raises(ValidationError) as exc_info:
            validate_csv(path)
        joined = " ".join(exc_info.value.violations)
        assert "runtime_sec" in joined

    def test_missing_multiple_columns(self, tmp_path):
        """CSV missing 3 required columns raises ValidationError listing all 3."""
        missing = ["runtime_sec", "cpu_cores", "memory_pressure"]
        cols = [c for c in REQUIRED_COLUMNS if c not in missing]
        path = str(tmp_path / "missing3.csv")
        with open(path, "w", newline="", encoding="utf-8") as fh:
            writer = csv.DictWriter(fh, fieldnames=cols)
            writer.writeheader()
            writer.writerows([{c: "1" for c in cols} for _ in range(600)])
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

    def test_exactly_500_rows_passes(self, tmp_path):
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
# memory_pressure tests
# ---------------------------------------------------------------------------

class TestMemoryPressure:
    def test_memory_pressure_above_one_raises(self, tmp_path):
        """memory_pressure=1.5 exceeds [0.0, 1.0] and raises ValidationError."""
        path = make_csv(tmp_path, overrides={"memory_pressure": ["1.5"]})
        with pytest.raises(ValidationError) as exc_info:
            validate_csv(path)
        assert any("memory_pressure" in v for v in exc_info.value.violations)

    def test_memory_pressure_below_zero_raises(self, tmp_path):
        """memory_pressure=-0.1 is below [0.0, 1.0] and raises ValidationError."""
        path = make_csv(tmp_path, overrides={"memory_pressure": ["-0.1"]})
        with pytest.raises(ValidationError) as exc_info:
            validate_csv(path)
        assert any("memory_pressure" in v for v in exc_info.value.violations)

    def test_memory_pressure_exactly_one_passes(self, tmp_path):
        """memory_pressure=1.0 is at the upper boundary and must pass."""
        path = make_csv(tmp_path, overrides={"memory_pressure": "1.0"})
        result = validate_csv(path)
        assert result["valid"] is True

    def test_memory_pressure_zero_passes(self, tmp_path):
        """memory_pressure=0.0 is at the lower boundary and must pass."""
        path = make_csv(tmp_path, overrides={"memory_pressure": "0.0"})
        result = validate_csv(path)
        assert result["valid"] is True


# ---------------------------------------------------------------------------
# workload_type tests
# ---------------------------------------------------------------------------

class TestWorkloadType:
    def test_invalid_workload_type_raises(self, tmp_path):
        """workload_type='INVALID' is not in VALID_WORKLOAD_TYPES and raises."""
        assert "INVALID" not in VALID_WORKLOAD_TYPES
        path = make_csv(tmp_path, overrides={"workload_type": ["INVALID"]})
        with pytest.raises(ValidationError) as exc_info:
            validate_csv(path)
        assert any("workload_type" in v for v in exc_info.value.violations)

    def test_valid_workload_types_pass(self, tmp_path):
        """All values in VALID_WORKLOAD_TYPES (ML, DB, WEB) pass validation."""
        path = make_csv(tmp_path)
        result = validate_csv(path)
        assert result["valid"] is True
        assert set(result["workload_type_counts"].keys()) == VALID_WORKLOAD_TYPES


# ---------------------------------------------------------------------------
# disk_type tests
# ---------------------------------------------------------------------------

class TestDiskType:
    def test_invalid_disk_type_raises(self, tmp_path):
        """disk_type='OPTANE' is not in VALID_DISK_TYPES and raises ValidationError."""
        assert "OPTANE" not in VALID_DISK_TYPES
        path = make_csv(tmp_path, overrides={"disk_type": ["OPTANE"]})
        with pytest.raises(ValidationError) as exc_info:
            validate_csv(path)
        assert any("disk_type" in v for v in exc_info.value.violations)

    def test_valid_disk_types_pass(self, tmp_path):
        """All values in VALID_DISK_TYPES (HDD, SSD, NVMe) appear and pass."""
        path = make_csv(tmp_path)
        result = validate_csv(path)
        assert result["valid"] is True


# ---------------------------------------------------------------------------
# cpu_avg_pct tests
# ---------------------------------------------------------------------------

class TestCpuAvgPct:
    def test_cpu_pct_above_100_raises(self, tmp_path):
        """cpu_avg_pct=101.0 exceeds [0.0, 100.0] and raises ValidationError."""
        path = make_csv(tmp_path, overrides={"cpu_avg_pct": ["101.0"]})
        with pytest.raises(ValidationError) as exc_info:
            validate_csv(path)
        assert any("cpu_avg_pct" in v for v in exc_info.value.violations)

    def test_cpu_pct_negative_raises(self, tmp_path):
        """cpu_avg_pct=-5.0 is below [0.0, 100.0] and raises ValidationError."""
        path = make_csv(tmp_path, overrides={"cpu_avg_pct": ["-5.0"]})
        with pytest.raises(ValidationError) as exc_info:
            validate_csv(path)
        assert any("cpu_avg_pct" in v for v in exc_info.value.violations)


# ---------------------------------------------------------------------------
# workload_complexity tests
# ---------------------------------------------------------------------------

class TestWorkloadComplexity:
    def test_complexity_out_of_range_raises(self, tmp_path):
        """workload_complexity=6 exceeds [1, 5] and raises ValidationError."""
        path = make_csv(tmp_path, overrides={"workload_complexity": ["6"]})
        with pytest.raises(ValidationError) as exc_info:
            validate_csv(path)
        assert any("workload_complexity" in v for v in exc_info.value.violations)

    def test_complexity_zero_raises(self, tmp_path):
        """workload_complexity=0 is below [1, 5] and raises ValidationError."""
        path = make_csv(tmp_path, overrides={"workload_complexity": ["0"]})
        with pytest.raises(ValidationError) as exc_info:
            validate_csv(path)
        assert any("workload_complexity" in v for v in exc_info.value.violations)


# ---------------------------------------------------------------------------
# Empty value tests
# ---------------------------------------------------------------------------

class TestEmptyValues:
    def test_empty_numeric_field_raises(self, tmp_path):
        """A row with an empty string in any NUMERIC_COLUMNS field raises ValidationError."""
        assert "cpu_cores" in NUMERIC_COLUMNS
        path = make_csv(tmp_path, overrides={"cpu_cores": [""]})
        with pytest.raises(ValidationError) as exc_info:
            validate_csv(path)
        assert any("cpu_cores" in v for v in exc_info.value.violations)


# ---------------------------------------------------------------------------
# Violation collection tests
# ---------------------------------------------------------------------------

class TestViolationCollection:
    def test_multiple_violations_collected(self, tmp_path):
        """All violations from multiple bad rows are collected, not just the first."""
        path = make_csv(
            tmp_path,
            overrides={"workload_type": ["BAD_1", "BAD_2", "BAD_3"]},
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
# Additional constraint tests (cpu_cores, disk_speed_class, non-numeric values)
# ---------------------------------------------------------------------------

class TestAdditionalConstraints:
    def test_cpu_cores_zero_raises(self, tmp_path):
        """cpu_cores=0 is not a positive integer and raises ValidationError."""
        path = make_csv(tmp_path, overrides={"cpu_cores": ["0"]})
        with pytest.raises(ValidationError) as exc_info:
            validate_csv(path)
        assert any("cpu_cores" in v for v in exc_info.value.violations)

    def test_cpu_cores_negative_raises(self, tmp_path):
        """cpu_cores=-1 is not a positive integer and raises ValidationError."""
        path = make_csv(tmp_path, overrides={"cpu_cores": ["-1"]})
        with pytest.raises(ValidationError) as exc_info:
            validate_csv(path)
        assert any("cpu_cores" in v for v in exc_info.value.violations)

    def test_disk_speed_class_invalid_raises(self, tmp_path):
        """disk_speed_class=4 is not in {1, 2, 3} and raises ValidationError."""
        path = make_csv(tmp_path, overrides={"disk_speed_class": ["4"]})
        with pytest.raises(ValidationError) as exc_info:
            validate_csv(path)
        assert any("disk_speed_class" in v for v in exc_info.value.violations)

    def test_non_numeric_value_in_numeric_column_raises(self, tmp_path):
        """A non-numeric string in a NUMERIC_COLUMNS field raises ValidationError."""
        path = make_csv(tmp_path, overrides={"cpu_avg_pct": ["not_a_number"]})
        with pytest.raises(ValidationError) as exc_info:
            validate_csv(path)
        assert any("cpu_avg_pct" in v for v in exc_info.value.violations)

    def test_violation_limit_caps_at_50(self, tmp_path):
        """When 50+ rows are bad, collection stops at 50 violations to save memory."""
        path = make_csv(tmp_path, overrides={"workload_type": ["BAD"] * 55})
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
        """Example row contains workload_type=ML and runtime_sec=14.37."""
        out = tmp_path / "template.csv"
        export_schema_template(str(out))
        with open(out, newline="", encoding="utf-8") as fh:
            reader = csv.DictReader(fh)
            row = list(reader)[0]
        assert row["workload_type"] == "ML"
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
            "valid_workload_types",
            "valid_disk_types",
            "min_row_count",
            "column_constraints",
        }
        assert set(info.keys()) == expected

    def test_schema_info_min_row_count(self):
        """get_schema_info reports min_row_count of 500."""
        info = get_schema_info()
        assert info["min_row_count"] == MIN_ROW_COUNT
        assert info["min_row_count"] == 500
