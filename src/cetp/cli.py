"""
CETP command-line interface.
Entry point: cetp (configured in pyproject.toml [project.scripts]).
"""
import json
import sys
from pathlib import Path
from typing import Optional

import click

from cetp import __version__


_DEFAULT_SLA = str(Path(__file__).parent.parent.parent / "sla_defaults.json")


@click.group()
@click.version_option(__version__, prog_name="cetp")
def main() -> None:
    """CETP — Cross-Environment Execution Time Prediction.

    Predicts production runtime from development environment metrics
    with SLA-aware flagging and SHAP-based explanations.
    """


# ---------------------------------------------------------------------------
# cetp predict
# ---------------------------------------------------------------------------

@main.command("predict")
@click.option("--workload-type", required=True, type=click.Choice(["ML", "DB", "WEB"]),
              help="Category of the workload (ML, DB, or WEB).")
@click.option("--workload-name", required=True, type=str,
              help="Identifier for the specific workload, e.g. ml_resnet, tpch_q3.")
@click.option("--complexity", required=True, type=click.IntRange(1, 5),
              help="Workload complexity level on a scale of 1 (trivial) to 5 (intensive).")
@click.option("--sla", "sla_path", default=_DEFAULT_SLA, show_default=True,
              type=click.Path(), help="Path to a JSON file with SLA threshold overrides.")
@click.option("--cpu-cores", type=int, default=None,
              help="Override: number of physical CPU cores.")
@click.option("--memory-gb", type=float, default=None,
              help="Override: total system RAM in GB.")
@click.option("--disk-type", type=click.Choice(["HDD", "SSD", "NVMe"]), default=None,
              help="Override: storage medium type.")
@click.option("--cpu-pct", type=float, default=None,
              help="Override: CPU utilisation percentage (0–100).")
@click.option("--mem-used-gb", type=float, default=None,
              help="Override: used memory in GB.")
@click.option("--fail-on-red", is_flag=True, default=False,
              help="Exit with code 1 if the SLA flag is RED.")
@click.option("--json", "output_json", is_flag=True, default=False,
              help="Output result as JSON instead of human-readable text.")
def predict(
    workload_type: str,
    workload_name: str,
    complexity: int,
    sla_path: str,
    cpu_cores: Optional[int],
    memory_gb: Optional[float],
    disk_type: Optional[str],
    cpu_pct: Optional[float],
    mem_used_gb: Optional[float],
    fail_on_red: bool,
    output_json: bool,
) -> None:
    """Predict production runtime for a workload.

    Collects the current machine's hardware profile, applies any manual
    overrides, and feeds the feature vector into the CETP prediction engine.
    Prints a point estimate, 90 % confidence interval, and an SLA flag.

    Example:

        cetp predict --workload-type ML --workload-name ml_resnet --complexity 3
    """
    from cetp.profiler import get_static_profile
    from cetp.predictor import CETPPredictor

    # Build feature dict from live profile + overrides
    profile = get_static_profile()
    feature_dict: dict = {
        "workload_type": workload_type,
        "workload_name": workload_name,
        "workload_complexity": complexity,
        "cpu_cores": cpu_cores if cpu_cores is not None else profile["cpu_cores"],
        "memory_total_gb": memory_gb if memory_gb is not None else profile["memory_total_gb"],
        "disk_type": disk_type if disk_type is not None else profile["disk_type"],
        "disk_speed_class": (
            profile["disk_speed_class"] if disk_type is None
            else _disk_speed_class(disk_type)
        ),
        "cpu_avg_pct": cpu_pct if cpu_pct is not None else 50.0,
        "memory_avg_gb": mem_used_gb if mem_used_gb is not None else profile["memory_total_gb"] * 0.5,
    }

    predictor = CETPPredictor(sla_config_path=sla_path)

    try:
        result = predictor.predict(feature_dict)
    except NotImplementedError as exc:
        click.echo(f"[cetp] Prediction engine not yet available: {exc}", err=True)
        click.echo(
            "[cetp] Run `cetp train --data <your_data.csv>` to create a model first.",
            err=True,
        )
        sys.exit(2)

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


def _disk_speed_class(disk_type: str) -> int:
    """Map disk type string to integer speed class."""
    return {"HDD": 1, "SSD": 2, "NVMe": 3}.get(disk_type, 2)


# ---------------------------------------------------------------------------
# cetp profile
# ---------------------------------------------------------------------------

@main.command("profile")
def profile() -> None:
    """Display current machine hardware specifications.

    Reads CPU, RAM, and disk characteristics from the local system
    using psutil and prints them in a human-readable table.

    Example:

        cetp profile
    """
    from cetp.profiler import get_static_profile

    hw = get_static_profile()
    click.echo("=== CETP System Profile ===")
    click.echo(f"  CPU cores (physical) : {hw['cpu_cores']}")
    click.echo(f"  Total RAM            : {hw['memory_total_gb']:.2f} GB")
    click.echo(f"  Disk type            : {hw['disk_type']}")
    click.echo(f"  Disk speed class     : {hw['disk_speed_class']}  (HDD=1, SSD=2, NVMe=3)")


# ---------------------------------------------------------------------------
# cetp train
# ---------------------------------------------------------------------------

@main.command("train")
@click.option("--data", "data_path", required=True, type=click.Path(exists=True),
              help="Path to the company CSV file containing historical run data.")
@click.option("--sla", "sla_path", default=None, type=click.Path(),
              help="Path to a company SLA JSON file (optional).")
@click.option("--output-dir", default=str(Path.home() / ".cetp"), show_default=True,
              type=click.Path(), help="Directory where the trained model will be saved.")
def train(data_path: str, sla_path: Optional[str], output_dir: str) -> None:
    """Train a custom BYOD model on company data.

    Validates the supplied CSV against the CETP schema, then fine-tunes
    a gradient-boosted model using warm-start hyperparameter transfer
    from the base model.

    Example:

        cetp train --data company_runs.csv --output-dir ~/.cetp
    """
    from cetp.trainer import CETTrainer

    click.echo(f"[cetp train] Loading data from: {data_path}")
    trainer = CETTrainer()

    try:
        result = trainer.train(csv_path=data_path, sla_path=sla_path, output_dir=output_dir)
        click.echo(f"[cetp train] Model saved to: {result['model_path']}")
        click.echo(f"[cetp train] R² = {result['r2_score']:.4f}  RMSE = {result['rmse']:.4f} s")
        for w in result.get("warnings", []):
            click.echo(f"[cetp train] WARNING: {w}", err=True)
    except FileNotFoundError as exc:
        click.echo(f"[cetp train] Error: {exc}", err=True)
        sys.exit(1)
    except NotImplementedError as exc:
        click.echo(f"[cetp train] Not yet implemented: {exc}", err=True)
        sys.exit(2)


# ---------------------------------------------------------------------------
# cetp validate
# ---------------------------------------------------------------------------

@main.command("validate")
@click.option("--data", "data_path", required=True, type=click.Path(exists=True),
              help="Path to the CSV file to validate against the CETP schema.")
def validate(data_path: str) -> None:
    """Validate a CSV file against the CETP schema.

    Checks column presence, data types, value ranges, and minimum row count.
    Prints all violations found — does not stop at the first error.

    Example:

        cetp validate --data company_runs.csv
    """
    from cetp.validator import validate_csv, ValidationError, InsufficientDataError

    click.echo(f"[cetp validate] Checking: {data_path}")
    try:
        result = validate_csv(data_path)
        click.echo(
            click.style(
                f"[cetp validate] PASSED — {result['row_count']} rows, no violations.",
                fg="green",
            )
        )
    except InsufficientDataError as exc:
        click.echo(click.style("[cetp validate] FAILED (insufficient data):", fg="red"), err=True)
        for v in exc.violations:
            click.echo(f"  • {v}", err=True)
        sys.exit(1)
    except ValidationError as exc:
        click.echo(click.style("[cetp validate] FAILED:", fg="red"), err=True)
        for v in exc.violations:
            click.echo(f"  • {v}", err=True)
        sys.exit(1)


# ---------------------------------------------------------------------------
# cetp schema
# ---------------------------------------------------------------------------

@main.command("schema")
@click.option("--export", "export_path", default=None, type=click.Path(),
              help="If provided, exports a blank template CSV with all required headers.")
def schema(export_path: Optional[str]) -> None:
    """Show or export the required CSV schema.

    Prints the list of required columns and their expected formats.
    Use --export to write a blank template CSV ready to fill in.

    Example:

        cetp schema
        cetp schema --export template.csv
    """
    from cetp.validator import REQUIRED_COLUMNS, export_schema_template

    click.echo("=== CETP Required CSV Schema ===")
    click.echo("  Minimum rows : 500")
    click.echo(f"  Columns ({len(REQUIRED_COLUMNS)}):")
    for col in REQUIRED_COLUMNS:
        click.echo(f"    • {col}")

    if export_path:
        export_schema_template(export_path)
        click.echo(f"\nTemplate exported to: {export_path}")


# ---------------------------------------------------------------------------
# cetp info
# ---------------------------------------------------------------------------

@main.command("info")
def info() -> None:
    """Show active model, version, and SLA thresholds.

    Reads ~/.cetp/ for a custom BYOD model, falls back to the base model,
    and displays the active SLA thresholds from sla_defaults.json.

    Example:

        cetp info
    """
    from cetp.predictor import CETPPredictor, CETP_DIR, BASE_MODEL_PATH

    click.echo(f"=== CETP Info (v{__version__}) ===")

    predictor = CETPPredictor()
    model_info = predictor.get_active_model_info()

    click.echo(f"  Model path     : {model_info['model_path']}")
    click.echo(f"  Model version  : {model_info['model_version']}")
    click.echo(f"  Trained on     : {model_info['trained_on']}")
    click.echo(f"  Feature count  : {model_info['feature_count']}")
    click.echo(f"  Custom model   : {'Yes' if model_info['custom_model'] else 'No'}")

    # Print SLA thresholds
    sla_path = Path(__file__).parent.parent.parent / "sla_defaults.json"
    if sla_path.exists():
        with open(sla_path) as fh:
            sla = json.load(fh)
        click.echo("\n  SLA Thresholds:")
        for wt, thresholds in sla.items():
            click.echo(
                f"    {wt:5s}  warn={thresholds['warn_at_sec']}s  "
                f"sla={thresholds['sla_runtime_sec']}s"
            )
    else:
        click.echo("  SLA config     : sla_defaults.json not found")

    if not BASE_MODEL_PATH.exists() and not (CETP_DIR / "model.pkl").exists():
        click.echo(
            click.style(
                "\n  [!] No model artefact found. Run `cetp train --data <file.csv>` first.",
                fg="yellow",
            )
        )
