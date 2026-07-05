"""
Per-machine calibration for `cetp measure`'s local benchmark duration.

Deliberately separate from workload_config.py's WORKLOAD_COMPLEXITY_MAP:
that table's (batch_size, num_iterations) pairs are baked into the trained
predictor as feature values (see predictor.py's predict()) and must never
change without retraining the model. This table only controls how many
iterations `cetp measure` actually runs locally to produce a real measured
runtime in a reasonable wall-clock window on THIS machine — it shares
batch_size with workload_config.py (a workload-definition property, not a
timing property) but calibrates num_iterations independently per machine.

Values below were derived by timing each model's real per-batch-size
throughput on this machine and solving for the iteration count that lands
each complexity level in the 90/105/120/180/210s target range (the same
target range used across every cloud tier during original data collection,
recomputed fresh here since per-iteration cost does not transfer across
hardware).

distilbert's batch=16 rate measured faster per-iteration than batch=8
(0.0435s vs 0.0689s) in this calibration run — non-monotonic, almost
certainly n=10 sampling noise at sub-100ms scale rather than a real effect.
Recorded as measured rather than smoothed.
"""

MEASURE_NUM_ITERATIONS = {
    "resnet18": {1: 615, 2: 223, 3: 150, 4: 226, 5: 140},
    "resnet50": {1: 335, 2: 51, 3: 34, 4: 51, 5: 26},
    "mobilenet": {1: 122, 2: 83, 3: 51, 4: 76, 5: 42},
    "distilbert": {1: 1306, 2: 2414, 3: 1594, 4: 2390, 5: 1424},
}


def get_measure_iterations(model_name: str, complexity_level: int) -> int:
    """
    Returns the locally-calibrated iteration count `cetp measure` should run
    on this machine. Raises ValueError for an unknown model/level, mirroring
    workload_config.get_workload_params().
    """
    if model_name not in MEASURE_NUM_ITERATIONS:
        raise ValueError(
            f"Invalid model '{model_name}', must be one of {list(MEASURE_NUM_ITERATIONS)}"
        )
    levels = MEASURE_NUM_ITERATIONS[model_name]
    if complexity_level not in levels:
        raise ValueError(
            f"Invalid complexity_level {complexity_level} for model '{model_name}', "
            f"must be one of {sorted(levels)}"
        )
    return levels[complexity_level]
