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
@click.option("--json", "output_json", is_flag=True, default=False,
              help="Output as JSON instead of formatted table.")
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
@click.option("--data", "data_path", required=True, type=click.Path(),
              help="Path to CSV file to validate.")
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

    wt_counts = result["workload_type_counts"]
    wt_str = ", ".join(f"{k}={v}" for k, v in sorted(wt_counts.items()))
    stats = result["runtime_stats"]
    click.echo("✓ Validation passed")
    click.echo(f"{'Rows':<14}: {result['row_count']}")
    click.echo(f"{'Workload types':<14}: {wt_str}")
    click.echo(
        f"{'Runtime range':<14}: {stats['min']:.2f}s — {stats['max']:.2f}s"
        f"  (mean: {stats['mean']:.2f}s)"
    )


# ---------------------------------------------------------------------------
# cetp schema
# ---------------------------------------------------------------------------

@main.command("schema")
@click.option("--export", "export_path", default=None, type=click.Path(),
              help="Export blank template CSV to this path.")
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
    for wt in ("ML", "DB", "WEB"):
        if wt in sla:
            t = sla[wt]
            click.echo(
                f"{wt:<4}: warn at {t['warn_at_sec']}s, SLA limit {t['sla_runtime_sec']}s"
            )


# ---------------------------------------------------------------------------
# cetp predict
# ---------------------------------------------------------------------------

@main.command("predict")
@click.option("--workload-type", required=True, type=click.Choice(["ML", "DB", "WEB"]),
              help="Workload class: ML, DB, or WEB.")
@click.option("--workload-name", required=True, type=str,
              help="Workload name e.g. ml_resnet, tpch_q3.")
@click.option("--complexity", required=True, type=click.IntRange(1, 5),
              help="Workload complexity level 1-5.")
@click.option("--sla", "sla_path", default=None, type=click.Path(),
              help="Path to SLA config JSON file.")
@click.option("--cpu-cores", type=int, default=None,
              help="Override: CPU core count.")
@click.option("--memory-gb", type=float, default=None,
              help="Override: total RAM in GB.")
@click.option("--disk-type", type=click.Choice(["HDD", "SSD", "NVMe"]), default=None,
              help="Override: disk type (HDD, SSD, NVMe).")
@click.option("--cpu-pct", type=float, default=None,
              help="Override: CPU utilisation percentage.")
@click.option("--mem-used-gb", type=float, default=None,
              help="Override: used memory in GB.")
@click.option("--fail-on-red", is_flag=True, default=False,
              help="Exit with code 1 if SLA flag is RED.")
@click.option("--json", "output_json", is_flag=True, default=False,
              help="Output result as JSON.")
def predict(
    workload_type: str,
    workload_name: str,
    complexity: int,
    sla_path: Optional[str],
    cpu_cores: Optional[int],
    memory_gb: Optional[float],
    disk_type: Optional[str],
    cpu_pct: Optional[float],
    mem_used_gb: Optional[float],
    fail_on_red: bool,
    output_json: bool,
) -> None:
    """Predict production runtime for a workload.

    Collects hardware profile, applies any manual overrides, and feeds
    the feature vector into the CETP prediction engine.

    Example:

        cetp predict --workload-type ML --workload-name ml_resnet --complexity 3
    """
    try:
        from cetp.predictor import CETPPredictor
    except ImportError as exc:
        click.echo(f"✗ Could not import predictor: {exc}")
        sys.exit(1)

    try:
        hw = get_static_profile()
    except Exception as exc:
        click.echo(f"✗ Error reading system profile: {exc}")
        sys.exit(1)

    _disk_map = {"HDD": 1, "SSD": 2, "NVMe": 3}
    feature_dict = {
        "workload_type": workload_type,
        "workload_name": workload_name,
        "workload_complexity": complexity,
        "cpu_cores": cpu_cores if cpu_cores is not None else hw["cpu_cores"],
        "memory_total_gb": memory_gb if memory_gb is not None else hw["memory_total_gb"],
        "disk_type": disk_type if disk_type is not None else hw["disk_type"],
        "disk_speed_class": (
            _disk_map.get(disk_type, 2) if disk_type is not None else hw["disk_speed_class"]
        ),
        "cpu_avg_pct": cpu_pct if cpu_pct is not None else 50.0,
        "memory_avg_gb": (
            mem_used_gb if mem_used_gb is not None else hw["memory_total_gb"] * 0.5
        ),
    }

    try:
        predictor = CETPPredictor(sla_config_path=sla_path)
        result = predictor.predict(feature_dict)
    except (NotImplementedError, FileNotFoundError):
        click.echo("✗ Model artefact not found.")
        click.echo("Run 'cetp train --data your_data.csv' to train a custom model,")
        click.echo("or add base_model.pkl to model/artifacts/.")
        sys.exit(1)
    except Exception as exc:
        click.echo(f"✗ Prediction failed: {exc}")
        sys.exit(1)

    if output_json:
        click.echo(json.dumps(result, indent=2))
    else:
        flag = result.get("sla_flag", "UNKNOWN")
        colour = {"GREEN": "green", "YELLOW": "yellow", "RED": "red"}.get(flag, "white")
        click.echo(f"Workload : {workload_name} ({workload_type}, complexity {complexity})")
        click.echo(
            f"Predicted: {result['predicted_sec']:.2f} s  "
            f"[90% CI {result['lower_sec']:.2f} – {result['upper_sec']:.2f} s]"
        )
        click.echo(f"SLA flag : {click.style(flag, fg=colour, bold=True)}")

    if fail_on_red and result.get("sla_flag") == "RED":
        sys.exit(1)


# ---------------------------------------------------------------------------
# cetp train
# ---------------------------------------------------------------------------

@main.command("train")
@click.option("--data", "data_path", required=True, type=click.Path(),
              help="Path to company CSV file.")
@click.option("--sla", "sla_path", default=None, type=click.Path(),
              help="Path to company SLA JSON file.")
@click.option("--output-dir", default=str(Path.home() / ".cetp"), show_default=True,
              type=click.Path(), help="Directory to save model.")
def train(data_path: str, sla_path: Optional[str], output_dir: str) -> None:
    """Train a BYOD custom model on company data.

    Validates the supplied CSV against the CETP schema, then fine-tunes
    a gradient-boosted model using warm-start hyperparameter transfer.

    Example:

        cetp train --data company_runs.csv --output-dir ~/.cetp
    """
    try:
        validate_csv(data_path)
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

    try:
        from cetp.trainer import CETTrainer
    except ImportError as exc:
        click.echo(f"✗ Could not import trainer: {exc}")
        sys.exit(1)

    try:
        trainer = CETTrainer()
        result = trainer.train(csv_path=data_path, sla_path=sla_path, output_dir=output_dir)
        click.echo(f"✓ Model saved to: {result['model_path']}")
        click.echo(f"  R² = {result['r2_score']:.4f}  RMSE = {result['rmse']:.4f} s")
        for w in result.get("warnings", []):
            click.echo(f"  WARNING: {w}")
    except (FileNotFoundError, NotImplementedError):
        click.echo("✗ Base hyperparameters not found.")
        click.echo("base_hyperparams.json must be present in model/artifacts/.")
        click.echo("This file is generated during base model training.")
        sys.exit(1)
    except Exception as exc:
        click.echo(f"✗ Training failed: {exc}")
        sys.exit(1)
