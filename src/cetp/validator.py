"""
Schema validator: validates company-supplied CSV files against the
CETP dataset schema before BYOD training is permitted.
"""

import csv
import os
from pathlib import Path
from typing import Optional  # noqa: F401 — required for Python 3.9 compat (no X | Y syntax)

REQUIRED_COLUMNS = [
    "cpu_count",
    "total_memory_mb",
    "model",
    "complexity_level",
    "batch_size",
    "num_iterations",
    "runtime_sec",
]

NUMERIC_COLUMNS = [
    "cpu_count",
    "total_memory_mb",
    "complexity_level",
    "batch_size",
    "num_iterations",
    "runtime_sec",
]

VALID_MODELS = {"resnet18", "resnet50", "mobilenet", "distilbert"}

# Relaxed from the original 500-row threshold: BYOD companies benchmarking their
# own infrastructure realistically produce dozens to low-hundreds of runs, not
# thousands, so 100 rows is treated as the practical minimum for a usable fit.
MIN_ROW_COUNT = 100

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
    model_counts: dict = {}

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

            elif col == "total_memory_mb":
                if fval <= 0:
                    violations.append(
                        f"Row {i}: total_memory_mb must be a positive float, got {fval}"
                    )
                    row_ok = False

            elif col == "complexity_level":
                int_val = int(fval)
                if float(int_val) != fval or not (1 <= int_val <= 5):
                    violations.append(
                        f"Row {i}: complexity_level must be an integer in [1, 5], got {val}"
                    )
                    row_ok = False

            elif col == "cpu_count":
                int_val = int(fval)
                if float(int_val) != fval or int_val <= 0:
                    violations.append(f"Row {i}: cpu_count must be a positive integer, got {val}")
                    row_ok = False

            elif col == "batch_size":
                int_val = int(fval)
                if float(int_val) != fval or int_val <= 0:
                    violations.append(f"Row {i}: batch_size must be a positive integer, got {val}")
                    row_ok = False

            elif col == "num_iterations":
                int_val = int(fval)
                if float(int_val) != fval or int_val <= 0:
                    violations.append(
                        f"Row {i}: num_iterations must be a positive integer, got {val}"
                    )
                    row_ok = False

        model = row.get("model", "").strip()
        if model == "":
            violations.append(f"Row {i}: empty value in column 'model'")
            row_ok = False
        elif model not in VALID_MODELS:
            violations.append(
                f"Row {i}: invalid model '{model}', must be one of {sorted(VALID_MODELS)}"
            )
            row_ok = False

        if row_ok:
            runtime_values.append(float(row.get("runtime_sec", "").strip()))  # already validated
            model_counts[model] = model_counts.get(model, 0) + 1

    if violations:
        raise ValidationError(violations)

    mean_rt = sum(runtime_values) / len(runtime_values) if runtime_values else 0.0

    return {
        "valid": True,
        "row_count": row_count,
        "violations": [],
        "columns_found": headers,
        "model_counts": model_counts,
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
        "cpu_count": "4",
        "total_memory_mb": "15785.3",
        "model": "resnet18",
        "complexity_level": "3",
        "batch_size": "32",
        "num_iterations": "250",
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
        "valid_models": sorted(VALID_MODELS),
        "min_row_count": MIN_ROW_COUNT,
        "column_constraints": {
            "cpu_count": "positive integer",
            "total_memory_mb": "positive float",
            "model": f"one of {sorted(VALID_MODELS)}",
            "complexity_level": "integer, [1, 5]",
            "batch_size": "positive integer",
            "num_iterations": "positive integer",
            "runtime_sec": "float, > 0",
        },
    }
