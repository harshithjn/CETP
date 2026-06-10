"""
Schema validator: validates company-supplied CSV files against the
CETP dataset schema before BYOD training is permitted.
"""
import csv
import os

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

VALID_WORKLOAD_TYPES = {"ML", "DB", "WEB"}
MIN_ROW_COUNT = 500

_NUMERIC_COLUMNS = {
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
}


class ValidationError(Exception):
    """Raised when CSV validation fails. Contains a list of all violations."""

    def __init__(self, violations: list) -> None:
        self.violations = violations
        super().__init__(f"{len(violations)} validation error(s) found")


class InsufficientDataError(ValidationError):
    """Raised when row count is below the minimum required for training."""
    pass


def validate_csv(filepath: str) -> dict:
    """
    Validates the CSV at filepath against the CETP schema.
    Collects ALL violations before raising — does not stop at first error.
    Returns a dict with keys: row_count, violations (list), valid (bool).
    Raises ValidationError if any violations are found.
    Raises InsufficientDataError if row count < MIN_ROW_COUNT.
    """
    violations: list[str] = []
    filepath = str(filepath)

    if not os.path.exists(filepath):
        raise FileNotFoundError(f"CSV file not found: {filepath}")

    with open(filepath, newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        headers = reader.fieldnames or []

        # Rule 1: all required columns present
        missing = [col for col in REQUIRED_COLUMNS if col not in headers]
        if missing:
            violations.append(f"Missing required columns: {missing}")
            # Cannot validate rows without the expected columns
            raise ValidationError(violations)

        rows = list(reader)

    row_count = len(rows)

    for i, row in enumerate(rows, start=2):  # start=2 because row 1 is header
        # Rule 5: no null/empty values in numeric columns
        for col in _NUMERIC_COLUMNS:
            val = row.get(col, "").strip()
            if val == "":
                violations.append(f"Row {i}: empty value in numeric column '{col}'")
                continue

            try:
                numeric_val = float(val)
            except ValueError:
                violations.append(f"Row {i}: non-numeric value '{val}' in column '{col}'")
                continue

            # Rule 2: runtime_sec must be > 0
            if col == "runtime_sec" and numeric_val <= 0:
                violations.append(
                    f"Row {i}: runtime_sec must be > 0, got {numeric_val}"
                )

            # Rule 3: memory_pressure must be in [0, 1]
            if col == "memory_pressure" and not (0.0 <= numeric_val <= 1.0):
                violations.append(
                    f"Row {i}: memory_pressure must be in [0, 1], got {numeric_val}"
                )

        # Rule 4: workload_type must be valid
        wt = row.get("workload_type", "").strip()
        if wt not in VALID_WORKLOAD_TYPES:
            violations.append(
                f"Row {i}: invalid workload_type '{wt}', must be one of {VALID_WORKLOAD_TYPES}"
            )

    # Rule 6: row count check (raise InsufficientDataError, a subclass of ValidationError)
    if row_count < MIN_ROW_COUNT:
        insufficient_violation = (
            f"Insufficient data: {row_count} rows found, minimum required is {MIN_ROW_COUNT}"
        )
        violations.append(insufficient_violation)
        raise InsufficientDataError(violations)

    if violations:
        raise ValidationError(violations)

    return {"row_count": row_count, "violations": [], "valid": True}


def export_schema_template(output_path: str) -> None:
    """
    Writes a blank CSV file with all required column headers to output_path.
    Adds one example row with clearly labelled placeholder values.
    """
    example_row = {
        "run_id": "run_001",
        "workload_type": "ML",
        "workload_name": "ml_resnet50",
        "workload_complexity": "3",
        "cpu_cores": "8",
        "memory_total_gb": "16.0",
        "cpu_avg_pct": "72.5",
        "effective_cpu": "2.2",
        "memory_avg_gb": "10.4",
        "memory_pressure": "0.65",
        "disk_read_mb": "120.0",
        "disk_write_mb": "45.0",
        "io_intensity": "18.3",
        "disk_type": "SSD",
        "disk_speed_class": "2",
        "runtime_sec": "14.7",
    }

    output_path = str(output_path)
    with open(output_path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=REQUIRED_COLUMNS)
        writer.writeheader()
        writer.writerow(example_row)
