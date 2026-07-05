"""
CETP command-line interface.
Entry point: cetp (configured in pyproject.toml [project.scripts]).
"""

from typing import Optional
import json
import sys
from pathlib import Path

import click

from cetp import __version__
from cetp.profiler import get_static_profile, build_feature_row  # noqa: F401
from cetp.validator import (
    validate_csv,
    export_schema_template,
    get_schema_info,
    ValidationError,
    InsufficientDataError,
)


@click.group()
@click.version_option(version=__version__, prog_name="cetp")
def main() -> None:
    """CETP — Cross-Environment Execution Time Prediction.

    Predict production runtime from development environment metrics
    with SLA-aware flagging and SHAP-based explanations.
    """


# ---------------------------------------------------------------------------
# cetp profile
# ---------------------------------------------------------------------------


@main.command("profile")
@click.option(
    "--json",
    "output_json",
    is_flag=True,
    default=False,
    help="Output as JSON instead of formatted table.",
)
def profile(output_json: bool) -> None:
    """Display current machine hardware specifications.

    Reads CPU, RAM, and disk characteristics from the local system
    and prints them in a formatted table.

    Example:

        cetp profile
        cetp profile --json
    """
    try:
        hw = get_static_profile()
    except Exception as exc:
        click.echo(f"Error reading system profile: {exc}")
        sys.exit(1)

    if output_json:
        click.echo(json.dumps(hw, indent=2))
    else:
        click.echo("=== CETP System Profile ===")
        click.echo(f"{'CPU cores':<21}: {hw['cpu_cores']}")
        click.echo(f"{'Total RAM':<21}: {hw['memory_total_gb']:.2f} GB")
        click.echo(f"{'Disk type':<21}: {hw['disk_type']}")
        click.echo(f"{'Disk speed class':<21}: {hw['disk_speed_class']}  (HDD=1, SSD=2, NVMe=3)")
        click.echo(f"{'Platform':<21}: {hw['platform']}")
        click.echo(f"{'Hostname':<21}: {hw['hostname']}")


# ---------------------------------------------------------------------------
# cetp validate
# ---------------------------------------------------------------------------


@main.command("validate")
@click.option(
    "--data", "data_path", required=True, type=click.Path(), help="Path to CSV file to validate."
)
def validate(data_path: str) -> None:
    """Validate a CSV file against the CETP schema.

    Checks column presence, data types, value ranges, and minimum row count.
    Prints all violations found — does not stop at the first error.

    Example:

        cetp validate --data company_runs.csv
    """
    try:
        result = validate_csv(data_path)
    except FileNotFoundError:
        click.echo(f"✗ File not found: {data_path}")
        sys.exit(1)
    except InsufficientDataError as exc:
        for v in exc.violations:
            click.echo(f"✗ {v}")
        sys.exit(1)
    except ValidationError as exc:
        for v in exc.violations:
            click.echo(f"✗ {v}")
        sys.exit(1)

    model_counts = result["model_counts"]
    model_str = ", ".join(f"{k}={v}" for k, v in sorted(model_counts.items()))
    stats = result["runtime_stats"]
    click.echo("✓ Validation passed")
    click.echo(f"{'Rows':<14}: {result['row_count']}")
    click.echo(f"{'Models':<14}: {model_str}")
    click.echo(
        f"{'Runtime range':<14}: {stats['min']:.2f}s — {stats['max']:.2f}s"
        f"  (mean: {stats['mean']:.2f}s)"
    )


# ---------------------------------------------------------------------------
# cetp schema
# ---------------------------------------------------------------------------


@main.command("schema")
@click.option(
    "--export",
    "export_path",
    default=None,
    type=click.Path(),
    help="Export blank template CSV to this path.",
)
def schema(export_path: Optional[str]) -> None:
    """Show or export the required CSV schema.

    Prints the list of required columns and their value constraints.
    Use --export to write a blank template CSV ready to fill in.

    Example:

        cetp schema
        cetp schema --export template.csv
    """
    info = get_schema_info()
    cols = info["required_columns"]
    constraints = info["column_constraints"]

    click.echo("=== CETP Dataset Schema ===")
    click.echo(f"Required columns ({len(cols)}):")
    for i in range(0, len(cols), 4):
        chunk = cols[i : i + 4]
        trailing = "," if i + 4 < len(cols) else ""
        click.echo(", ".join(chunk) + trailing)
    click.echo("Constraints:")
    for col_name, constraint in constraints.items():
        click.echo(f"  {col_name:<19}: {constraint}")
    click.echo(f"Minimum rows required: {info['min_row_count']}")

    if export_path:
        try:
            export_schema_template(export_path)
            click.echo(f"✓ Template exported to {Path(export_path).resolve()}")
        except Exception as exc:
            click.echo(f"✗ Export failed: {exc}")
            sys.exit(1)


# ---------------------------------------------------------------------------
# cetp info
# ---------------------------------------------------------------------------


@main.command("info")
def info() -> None:
    """Show active model, version, and SLA thresholds.

    Checks for custom and base model artefacts, then displays
    the active SLA thresholds.

    Example:

        cetp info
    """
    click.echo("=== CETP Info ===")
    click.echo(f"{'Version':<13}: {__version__}")

    custom_model = Path.home() / ".cetp" / "custom_model.pkl"
    base_model = Path(__file__).parent.parent.parent / "model" / "artifacts" / "base_model.pkl"

    if custom_model.exists():
        model_status = "Custom model active"
    elif base_model.exists():
        model_status = "Base model active"
    else:
        model_status = "No model artefact found — run cetp train or add base_model.pkl"

    click.echo(f"{'Model status':<13}: {model_status}")

    user_sla = Path.home() / ".cetp" / "sla.json"
    pkg_sla = Path(__file__).parent.parent.parent / "sla_defaults.json"

    sla_path = None
    if user_sla.exists():
        sla_path = user_sla
    elif pkg_sla.exists():
        sla_path = pkg_sla

    if sla_path is None:
        click.echo("No SLA config found")
        return

    try:
        with open(sla_path) as fh:
            sla = json.load(fh)
    except Exception as exc:
        click.echo(f"Error loading SLA config: {exc}")
        return

    click.echo("=== SLA Thresholds ===")
    for model_name in sorted(sla.keys()):
        t = sla[model_name]
        click.echo(
            f"{model_name:<11}: warn at {t['warn_at_sec']}s, SLA limit {t['sla_runtime_sec']}s"
        )


# ---------------------------------------------------------------------------
# cetp predict
# ---------------------------------------------------------------------------


def _friendly_shap_name(raw: str) -> str:
    """Strip ColumnTransformer prefixes so SHAP feature names read naturally."""
    if raw.startswith("remainder__"):
        return raw[len("remainder__") :]
    if raw.startswith("cat__model_"):
        return f"model={raw[len('cat__model_') :]}"
    return raw


def _direction_label(shap_value: float, magnitude_threshold: float) -> str:
    """
    Presentation-only heuristic: a SHAP value tiny relative to the predicted
    runtime reads as noise to a human, not a real driver, so it's labelled
    "minimal effect" rather than a misleadingly confident increase/decrease.
    """
    if abs(shap_value) < magnitude_threshold:
        return "minimal effect"
    return "increases runtime" if shap_value > 0 else "decreases runtime"


@main.command("predict")
@click.option(
    "--model",
    "model_name",
    required=True,
    type=click.Choice(["resnet18", "resnet50", "mobilenet", "distilbert"]),
    help="Which ML model workload to predict runtime for.",
)
@click.option(
    "--complexity",
    required=True,
    type=click.IntRange(1, 5),
    help="Workload complexity level, 1 (lightest) to 5 (heaviest).",
)
@click.option(
    "--cpu-cores",
    type=int,
    default=None,
    help="Target machine CPU core count. If omitted, auto-profiles this machine.",
)
@click.option(
    "--memory-gb",
    type=float,
    default=None,
    help="Target machine RAM in GB. If omitted, auto-profiles this machine.",
)
@click.option(
    "--sla",
    "sla_path",
    type=click.Path(exists=True),
    default=None,
    help="Path to custom sla.json. Uses package defaults if omitted.",
)
@click.option(
    "--fail-on-red", is_flag=True, help="Exit with code 1 if SLA flag is RED (for CI/CD use)."
)
@click.option("--json", "as_json", is_flag=True, help="Output as JSON.")
def predict(
    model_name: str,
    complexity: int,
    cpu_cores: Optional[int],
    memory_gb: Optional[float],
    sla_path: Optional[str],
    fail_on_red: bool,
    as_json: bool,
) -> None:
    """Predict runtime for an ML inference workload on target hardware.

    If --cpu-cores and --memory-gb are omitted, profiles the CURRENT machine.

    Example:

        cetp predict --model resnet18 --complexity 3 --cpu-cores 4 --memory-gb 8
    """
    try:
        from cetp.predictor import CETPPredictor
    except ImportError as exc:
        click.echo(f"✗ Could not import predictor: {exc}")
        sys.exit(1)

    if cpu_cores is None or memory_gb is None:
        try:
            hw = get_static_profile()
        except Exception as exc:
            click.echo(f"✗ Error reading system profile: {exc}")
            sys.exit(1)
        cpu_count = cpu_cores if cpu_cores is not None else hw["cpu_cores"]
        memory_gb_effective = memory_gb if memory_gb is not None else hw["memory_total_gb"]
    else:
        cpu_count = cpu_cores
        memory_gb_effective = memory_gb

    # ×1024, not ×1000: psutil.virtual_memory().total-derived values (profiler.py,
    # and thus the training data itself) use binary GB, so this must match.
    total_memory_mb = memory_gb_effective * 1024.0

    try:
        predictor = CETPPredictor(sla_config_path=sla_path)
    except FileNotFoundError as exc:
        click.echo("✗ Model artefact not found.")
        click.echo(str(exc))
        click.echo("Run 'cetp train --data your_data.csv' to train a custom model.")
        sys.exit(1)

    try:
        result = predictor.predict(model_name, complexity, cpu_count, total_memory_mb)
    except FileNotFoundError as exc:
        click.echo("✗ Model artefact not found.")
        click.echo(str(exc))
        sys.exit(1)
    except (ValueError, RuntimeError) as exc:
        click.echo(f"✗ Prediction failed: {exc}")
        sys.exit(1)
    except Exception as exc:
        click.echo(f"✗ Prediction failed: {exc}")
        sys.exit(1)

    upper = result["confidence_interval"][1]
    sla_limit = None
    try:
        sla_flag = predictor.compute_sla_flag(upper, model_name)
        sla_limit = predictor._sla_config[model_name]["sla_runtime_sec"]
        sla_line = f"{sla_flag} (limit: {sla_limit}s)"
    except KeyError:
        sla_flag = None
        sla_line = "UNKNOWN (no SLA config for this model)"

    shap_explanation = predictor.get_shap_explanation(
        model_name, complexity, cpu_count, total_memory_mb
    )
    shap_features = shap_explanation["features"]

    if as_json:
        output = dict(result)
        output["sla_flag"] = sla_flag
        output["sla_threshold_sec"] = sla_limit
        output["shap_base_value"] = shap_explanation["base_value"]
        output["top_shap_features"] = shap_features
        click.echo(json.dumps(output, indent=2))
    else:
        if result["confidence_status"] == "EXTRAPOLATED":
            click.echo(
                click.style(
                    "⚠ WARNING: This hardware configuration is outside the range this "
                    "model was validated on. Prediction accuracy is not guaranteed.",
                    fg="yellow",
                    bold=True,
                )
            )
            for w in result["confidence_warnings"]:
                click.echo(f"  - {w}")
            click.echo()

        lower, upper_ci = result["confidence_interval"]
        click.echo("=== Prediction Result ===")
        click.echo(f"Model: {model_name} (complexity {complexity})")
        click.echo(f"Predicted runtime: {result['predicted_runtime_sec']:.1f}s")
        click.echo(f"Confidence interval: [{lower:.1f}s, {upper_ci:.1f}s] (90%)")
        click.echo(f"Confidence status: {result['confidence_status']}")
        click.echo(f"SLA status: {sla_line}")

        if shap_features:
            click.echo()
            click.echo("Top factors:")
            magnitude_threshold = max(0.01 * abs(result["predicted_runtime_sec"]), 0.5)
            name_width = max(len(_friendly_shap_name(f["feature"])) for f in shap_features)
            for f in shap_features:
                name = _friendly_shap_name(f["feature"])
                label = _direction_label(f["shap_value"], magnitude_threshold)
                click.echo(f"  {name:<{name_width}}  -> {label}")

    if fail_on_red and sla_flag == "RED":
        sys.exit(1)


# ---------------------------------------------------------------------------
# cetp measure
# ---------------------------------------------------------------------------


def _run_local_measurement(workload_fn, batch_size: int, num_iterations: int) -> dict:
    """
    Times a workload while sampling CPU/memory, using the same primitives
    build_feature_row() composes for dataset collection (take_snapshot,
    RuntimeSampler, take_end_snapshot). Not build_feature_row() itself: that
    function's signature is CSV-row bookkeeping (workload_type, workload_name,
    run_id, batch_id) irrelevant to a single ad hoc local measurement, and it
    takes a zero-argument callable where ours needs (batch_size, num_iterations).
    profiler.py has no standalone measure() — this composes the same pieces
    build_feature_row() does, directly.
    """
    from cetp.profiler import RuntimeSampler, take_snapshot, take_end_snapshot

    sampler = RuntimeSampler(interval_sec=0.5)
    start_snap = take_snapshot()
    sampler.start()
    try:
        workload_fn(batch_size, num_iterations)
    finally:
        sampler.stop()
    _read_mb, _write_mb, runtime_sec = take_end_snapshot(start_snap)
    averages = sampler.get_averages()

    return {
        "runtime_sec": runtime_sec,
        # averages are computed from psutil bytes/1e9 (profiler.py's own basis,
        # see RuntimeSampler._sample_loop) — x1000 here matches that same basis.
        "peak_memory_mb": round(averages["memory_peak_gb"] * 1000, 1),
        "avg_cpu_pct": averages["cpu_avg_pct"],
    }


@main.command("measure")
@click.option(
    "--model",
    "model_name",
    required=True,
    type=click.Choice(["resnet18", "resnet50", "mobilenet", "distilbert"]),
    help="Which reference workload to run locally.",
)
@click.option(
    "--complexity",
    required=True,
    type=click.IntRange(1, 5),
    help="Workload complexity level, 1 (lightest) to 5 (heaviest).",
)
@click.option("--json", "as_json", is_flag=True, help="Output as JSON.")
def measure(model_name: str, complexity: int, as_json: bool) -> None:
    """Actually run the reference benchmark for a model/complexity LOCALLY.

    Runs the real torch/torchvision/transformers inference workload on this
    machine (no predictor, no trained model involved) and reports the real
    measured runtime, peak memory, and average CPU utilization.

    Example:

        cetp measure --model resnet18 --complexity 1 --json
    """
    from cetp.workload_config import get_workload_params
    from cetp.measure_calibration import get_measure_iterations
    from cetp.workloads import WORKLOAD_FUNCTIONS

    # batch_size is shared with `predict` (a workload-definition property);
    # num_iterations is NOT — it's calibrated per-machine for a reasonable
    # local wall-clock duration and must stay independent of the trained
    # predictor's feature space (see measure_calibration.py's docstring).
    batch_size, _ = get_workload_params(model_name, complexity)
    num_iterations = get_measure_iterations(model_name, complexity)
    fn = WORKLOAD_FUNCTIONS[model_name]

    click.echo(
        f"Running {model_name} at complexity {complexity} "
        f"(batch={batch_size}, iterations={num_iterations})... "
        f"this may take 60-400 seconds."
    )

    result = _run_local_measurement(fn, batch_size, num_iterations)

    output = {
        "measured_runtime_sec": result["runtime_sec"],
        "measured_peak_memory_mb": result["peak_memory_mb"],
        "measured_avg_cpu_pct": result["avg_cpu_pct"],
        "model_used": model_name,
        "complexity_level": complexity,
        "batch_size": batch_size,
        "num_iterations": num_iterations,
    }

    if as_json:
        click.echo(json.dumps(output, indent=2))
    else:
        click.echo(f"Measured runtime: {result['runtime_sec']:.1f}s")
        click.echo(f"Peak memory: {result['peak_memory_mb']:.0f}MB")
        click.echo(f"Avg CPU: {result['avg_cpu_pct']:.0f}%")


# ---------------------------------------------------------------------------
# cetp train
# ---------------------------------------------------------------------------


@main.command("train")
@click.option(
    "--data",
    "csv_path",
    required=True,
    type=click.Path(exists=True),
    help="Path to company CSV file.",
)
@click.option(
    "--sla",
    "sla_path",
    default=None,
    type=click.Path(exists=True),
    help="Path to company SLA JSON file.",
)
@click.option(
    "--output-dir",
    default=None,
    type=click.Path(),
    help="Directory to save model. Defaults to ~/.cetp/.",
)
def train(csv_path: str, sla_path: Optional[str], output_dir: Optional[str]) -> None:
    """Train a BYOD custom model on your own benchmark data.

    Validates the supplied CSV against the CETP schema, then fits a
    gradient-boosted model using the same proven hyperparameters as the
    base model.

    Example:

        cetp train --data company_runs.csv --output-dir ~/.cetp
    """
    try:
        from cetp.trainer import CETPTrainer
    except ImportError as exc:
        click.echo(f"✗ Could not import trainer: {exc}")
        sys.exit(1)

    trainer = CETPTrainer(output_dir=output_dir)

    try:
        result = trainer.train(csv_path, sla_path)
    except ValidationError:
        # trainer.train() already prints every violation before re-raising.
        sys.exit(1)
    except Exception as exc:
        click.echo(f"✗ Training failed: {exc}")
        sys.exit(1)

    metrics = result["metrics"]
    click.echo(f"✓ Model saved to: {result['model_path']}")
    click.echo(
        f"  R²={metrics['r2']:.4f}  RMSE={metrics['rmse']:.4f}s  "
        f"MAE={metrics['mae']:.4f}s  MAPE={metrics['mape']:.2f}%"
    )
    click.echo(f"  Rows used     : {result['row_count']}")
    click.echo(f"  Training range: {result['training_range_path']}")
    click.echo(f"  Metadata      : {result['metadata_path']}")
    if result["low_accuracy_warning"]:
        click.echo(f"  ⚠ R²={metrics['r2']:.4f} is below the 0.80 accuracy threshold.")
