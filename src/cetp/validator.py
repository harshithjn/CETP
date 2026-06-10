"""
Schema validator: validates company-supplied CSV files against the
CETP dataset schema before BYOD training is permitted.
"""

import csv
import os
from pathlib import Path
from typing import Optional  # noqa: F401 — required for Python 3.9 compat (no X | Y syntax)

REQUIRED_COLUMNS = [
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

NUMERIC_COLUMNS = [
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
    "disk_speed_class",
    "runtime_sec",
]

VALID_WORKLOAD_TYPES = {"ML", "DB", "WEB"}
VALID_DISK_TYPES = {"HDD", "SSD", "NVMe"}
MIN_ROW_COUNT = 500

_MAX_ROW_VIOLATIONS = 50


class ValidationError(Exception):
    """
    Raised when CSV validation fails.
    Always contains self.violations: list of strings describing every problem found.
    Never stops at the first error — collects ALL violations before raising.
    """

    def __init__(self, violations: list) -> None:
        self.violations = violations
        super().__init__(
            f"{len(violations)} validation error(s) found:\n"
            + "\n".join(f"  - {v}" for v in violations)
        )


class InsufficientDataError(ValidationError):
    """
    Raised specifically when row count is below MIN_ROW_COUNT.
    Subclass of ValidationError so callers can catch either.
    """

    pass


def validate_csv(filepath: str) -> dict:
    """
    Validates the CSV at filepath against the CETP schema.
    Collects ALL violations before raising — does not stop at first error.
    Raises FileNotFoundError if the file does not exist.
    Raises ValidationError if any schema violations are found.
    Raises InsufficientDataError if row count < MIN_ROW_COUNT.
    Returns a result dict when valid.
    """
    violations: list = []
    filepath = str(filepath)

    if not os.path.exists(filepath):
        raise FileNotFoundError(f"CSV file not found: {filepath}")

    with open(filepath, newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        headers: list = list(reader.fieldnames or [])

        missing = [col for col in REQUIRED_COLUMNS if col not in headers]
        if missing:
            raise ValidationError([f"Missing required columns: {missing}"])

        rows = list(reader)

    row_count = len(rows)

    if row_count < MIN_ROW_COUNT:
        raise InsufficientDataError(
            [f"Insufficient data: {row_count} rows found, minimum required is {MIN_ROW_COUNT}"]
        )

    runtime_values: list = []
    workload_type_counts: dict = {}

    for i, row in enumerate(rows, start=2):
        if len(violations) >= _MAX_ROW_VIOLATIONS:
            break

        row_ok = True

        for col in NUMERIC_COLUMNS:
            val = row.get(col, "").strip()

            if val == "":
                violations.append(f"Row {i}: empty value in column '{col}'")
                row_ok = False
                continue

            try:
                fval = float(val)
            except ValueError:
                violations.append(f"Row {i}: non-numeric value '{val}' in column '{col}'")
                row_ok = False
                continue

            if col == "runtime_sec":
                if fval <= 0:
                    violations.append(f"Row {i}: runtime_sec must be > 0, got {fval}")
                    row_ok = False

            elif col == "memory_pressure":
                if not (0.0 <= fval <= 1.0):
                    violations.append(f"Row {i}: memory_pressure must be in [0.0, 1.0], got {fval}")
                    row_ok = False

            elif col == "cpu_avg_pct":
                if not (0.0 <= fval <= 100.0):
                    violations.append(f"Row {i}: cpu_avg_pct must be in [0.0, 100.0], got {fval}")
                    row_ok = False

            elif col == "workload_complexity":
                int_val = int(fval)
                if float(int_val) != fval or not (1 <= int_val <= 5):
                    violations.append(
                        f"Row {i}: workload_complexity must be an integer in [1, 5], got {val}"
                    )
                    row_ok = False

            elif col == "cpu_cores":
                int_val = int(fval)
                if float(int_val) != fval or int_val <= 0:
                    violations.append(f"Row {i}: cpu_cores must be a positive integer, got {val}")
                    row_ok = False

            elif col == "disk_speed_class":
                int_val = int(fval)
                if float(int_val) != fval or int_val not in {1, 2, 3}:
                    violations.append(
                        f"Row {i}: disk_speed_class must be one of 1, 2, 3, got {val}"
                    )
                    row_ok = False

        wt = row.get("workload_type", "").strip()
        if wt not in VALID_WORKLOAD_TYPES:
            violations.append(
                f"Row {i}: invalid workload_type '{wt}', "
                f"must be one of {sorted(VALID_WORKLOAD_TYPES)}"
            )
            row_ok = False

        dt = row.get("disk_type", "").strip()
        if dt not in VALID_DISK_TYPES:
            violations.append(
                f"Row {i}: invalid disk_type '{dt}', must be one of {sorted(VALID_DISK_TYPES)}"
            )
            row_ok = False

        if row_ok:
            runtime_values.append(float(row.get("runtime_sec", "").strip()))  # already validated
            workload_type_counts[wt] = workload_type_counts.get(wt, 0) + 1

    if violations:
        raise ValidationError(violations)

    mean_rt = sum(runtime_values) / len(runtime_values) if runtime_values else 0.0

    return {
        "valid": True,
        "row_count": row_count,
        "violations": [],
        "columns_found": headers,
        "workload_type_counts": workload_type_counts,
        "runtime_stats": {
            "min": min(runtime_values) if runtime_values else 0.0,
            "max": max(runtime_values) if runtime_values else 0.0,
            "mean": mean_rt,
        },
    }


def export_schema_template(output_path: str) -> None:
    """
    Writes a CSV file to output_path with all required column headers and
    one example row with realistic placeholder values.
    """
    example_row = {
        "run_id": "0",
        "workload_type": "ML",
        "workload_name": "ml_resnet",
        "workload_complexity": "3",
        "cpu_cores": "8",
        "memory_total_gb": "16.0",
        "cpu_avg_pct": "45.2",
        "effective_cpu": "4.384",
        "memory_avg_gb": "6.4",
        "memory_pressure": "0.4",
        "disk_read_mb": "120.5",
        "disk_write_mb": "34.2",
        "io_intensity": "10.75",
        "disk_type": "SSD",
        "disk_speed_class": "2",
        "runtime_sec": "14.37",
    }

    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=REQUIRED_COLUMNS)
        writer.writeheader()
        writer.writerow(example_row)


def get_schema_info() -> dict:
    """Returns a dict describing the CETP schema for documentation purposes."""
    return {
        "required_columns": REQUIRED_COLUMNS,
        "numeric_columns": NUMERIC_COLUMNS,
        "valid_workload_types": list(VALID_WORKLOAD_TYPES),
        "valid_disk_types": list(VALID_DISK_TYPES),
        "min_row_count": MIN_ROW_COUNT,
        "column_constraints": {
            "runtime_sec": "float, > 0",
            "memory_pressure": "float, [0.0, 1.0]",
            "workload_type": "one of ML, DB, WEB",
            "disk_type": "one of HDD, SSD, NVMe",
            "workload_complexity": "integer, [1, 5]",
            "cpu_cores": "positive integer",
            "cpu_avg_pct": "float, [0.0, 100.0]",
            "disk_speed_class": "integer, one of 1, 2, 3",
        },
    }
