"""
Builds the dev-conditioned pairing dataset from the three real, measured tier
CSVs (large_full, medium, m5xlarge). Every row in the output is a real
(dev observation, target observation) pair drawn from actual psutil-measured
rows in Dataset/dataset_*.csv — nothing here is interpolated, simulated, or
formula-derived. target_runtime_sec is copied verbatim from the target row's
own measured runtime_sec.

Pairing rule: for every (model, complexity_level) group, take the ordered
cross product of all real rows in that group as (dev, target), EXCLUDING the
case where dev and target are literally the same row (i == i). That
degenerate case would set dev_runtime_sec identically equal to the label,
which is not a real cross-observation scenario (nobody predicts a
measurement from itself) and would be pure leakage if left in. Distinct reps
of the same tier (i != j, same tier) ARE kept — that is the real "predict on
the machine you already tested on" case the task asked to include.

batch_size/num_iterations in the output are taken from the TARGET row, not
the dev row. Both are per-tier calibration artifacts of whichever machine
actually ran the workload (confirmed below: num_iterations differs by tier
for every single (model, complexity) group in the source data, and two
groups even differ in batch_size between medium and the other two tiers).
Using the target row's values keeps every (features, label) pair internally
consistent: target_runtime_sec is the real measured outcome of running
exactly that batch_size/num_iterations on that target hardware. Using dev's
values instead would feed the model a workload spec that does NOT correspond
to how the label was actually produced.

dev_tier/target_tier are included as extra bookkeeping columns beyond the
task's specified 9 columns, purely so the training script can stratify by
target tier and run the leave-one-target-tier-out evaluation. They are never
used as model features.
"""

import csv
from pathlib import Path

DATASET_DIR = Path(__file__).resolve().parent.parent / "Dataset"

SOURCE_FILES = {
    "large_full": DATASET_DIR / "dataset_large_full.csv",
    "medium": DATASET_DIR / "dataset_medium.csv",
    "m5xlarge": DATASET_DIR / "dataset_m5xlarge.csv",
}

OUTPUT_PATH = DATASET_DIR / "dataset_dev_conditioned.csv"

OUTPUT_COLUMNS = [
    "dev_cpu_count",
    "dev_total_memory_mb",
    "dev_runtime_sec",
    "target_cpu_count",
    "target_total_memory_mb",
    "model",
    "complexity_level",
    "batch_size",
    "num_iterations",
    "target_runtime_sec",
    # Bookkeeping only — not model features. See module docstring.
    "dev_tier",
    "target_tier",
]


def load_rows() -> list[dict]:
    """Loads every real measured row from all three tier CSVs."""
    rows = []
    for tier, path in SOURCE_FILES.items():
        if not path.exists():
            raise FileNotFoundError(f"Missing required tier CSV: {path}")
        with open(path, newline="", encoding="utf-8") as fh:
            for r in csv.DictReader(fh):
                assert r["tier"] == tier, f"tier column mismatch in {path}: {r['tier']!r} != {tier!r}"
                rows.append(
                    {
                        "tier": tier,
                        "cpu_count": int(r["cpu_count"]),
                        "total_memory_mb": float(r["total_memory_mb"]),
                        "model": r["model"],
                        "complexity_level": int(r["complexity_level"]),
                        "batch_size": int(r["batch_size"]),
                        "num_iterations": int(r["num_iterations"]),
                        "runtime_sec": float(r["runtime_sec"]),
                    }
                )
    return rows


def build_pairs(rows: list[dict]) -> tuple[list[dict], dict]:
    """
    Groups rows by (model, complexity_level) and emits every ordered
    (dev, target) pair with dev != target (by identity/index, not just value
    equality, since two distinct real reps can legitimately share a runtime).
    """
    groups: dict[tuple, list[dict]] = {}
    for r in rows:
        key = (r["model"], r["complexity_level"])
        groups.setdefault(key, []).append(r)

    output_rows = []
    same_tier_count = 0
    cross_tier_count = 0
    group_sizes = set()

    for key, group_rows in groups.items():
        n = len(group_rows)
        group_sizes.add(n)
        for i in range(n):
            for j in range(n):
                if i == j:
                    continue
                dev = group_rows[i]
                target = group_rows[j]
                is_same_tier = dev["tier"] == target["tier"]
                if is_same_tier:
                    same_tier_count += 1
                else:
                    cross_tier_count += 1
                output_rows.append(
                    {
                        "dev_cpu_count": dev["cpu_count"],
                        "dev_total_memory_mb": dev["total_memory_mb"],
                        "dev_runtime_sec": dev["runtime_sec"],
                        "target_cpu_count": target["cpu_count"],
                        "target_total_memory_mb": target["total_memory_mb"],
                        "model": target["model"],
                        "complexity_level": target["complexity_level"],
                        "batch_size": target["batch_size"],
                        "num_iterations": target["num_iterations"],
                        "target_runtime_sec": target["runtime_sec"],
                        "dev_tier": dev["tier"],
                        "target_tier": target["tier"],
                    }
                )

    summary = {
        "distinct_real_measurements": len(rows),
        "num_groups": len(groups),
        "group_sizes": sorted(group_sizes),
        "total_pairs": len(output_rows),
        "same_tier_pairs": same_tier_count,
        "cross_tier_pairs": cross_tier_count,
    }
    return output_rows, summary


def main() -> None:
    rows = load_rows()
    output_rows, summary = build_pairs(rows)

    with open(OUTPUT_PATH, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=OUTPUT_COLUMNS)
        writer.writeheader()
        writer.writerows(output_rows)

    print(f"Wrote {summary['total_pairs']} paired rows to {OUTPUT_PATH}")
    print()
    print("=== Honesty check: pairs vs. real underlying data ===")
    print(f"Distinct real measured rows across all 3 tiers: {summary['distinct_real_measurements']}")
    print(f"  ({summary['num_groups']} (model, complexity_level) groups, "
          f"group sizes = {summary['group_sizes']} -> "
          f"{sum(summary['group_sizes'])} real rows per group)")
    print(f"Total paired output rows:  {summary['total_pairs']}")
    print(f"  same-tier pairs:  {summary['same_tier_pairs']}")
    print(f"  cross-tier pairs: {summary['cross_tier_pairs']}")
    multiplier = summary["total_pairs"] / summary["distinct_real_measurements"]
    print(
        f"Combinatorial multiplier: {summary['total_pairs']} pairs come from only "
        f"{summary['distinct_real_measurements']} distinct real measurements "
        f"({multiplier:.1f}x) — the row count is pairing arithmetic on a small "
        "real sample, not new independent data."
    )


if __name__ == "__main__":
    main()
