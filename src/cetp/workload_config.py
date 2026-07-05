"""
Canonical complexity-to-workload mapping, derived from real calibration data
on the 'large' reference tier.

These (batch_size, num_iterations) pairs are baked into the trained model as
feature values (see predictor.py's predict(), which passes them straight into
the row fed to the fitted pipeline) — they must stay exactly as they were
during original data collection, or every `cetp predict` prediction silently
drifts for a given complexity_level without the model being retrained. This
map is NOT the place to encode per-machine local benchmark timing; see
measure_calibration.py for that (used only by `cetp measure`).
"""

WORKLOAD_COMPLEXITY_MAP = {
    "resnet18": {1: (8, 310), 2: (16, 280), 3: (32, 250), 4: (32, 380), 5: (64, 340)},
    "resnet50": {1: (8, 125), 2: (16, 115), 3: (32, 100), 4: (32, 150), 5: (64, 135)},
    "mobilenet": {1: (16, 235), 2: (32, 210), 3: (64, 190), 4: (64, 285), 5: (128, 255)},
    "distilbert": {1: (8, 973), 2: (16, 631), 3: (32, 390), 4: (32, 566), 5: (64, 352)},
}
# Each tuple is (batch_size, num_iterations)

VALID_MODELS = list(WORKLOAD_COMPLEXITY_MAP.keys())


def get_workload_params(model_name: str, complexity_level: int) -> tuple:
    """
    Returns (batch_size, num_iterations) for a given model and complexity level.
    Raises ValueError with a clear message if model_name or complexity_level is invalid.
    """
    if model_name not in WORKLOAD_COMPLEXITY_MAP:
        raise ValueError(f"Invalid model '{model_name}', must be one of {VALID_MODELS}")

    levels = WORKLOAD_COMPLEXITY_MAP[model_name]
    if complexity_level not in levels:
        valid_levels = sorted(levels.keys())
        raise ValueError(
            f"Invalid complexity_level {complexity_level} for model '{model_name}', "
            f"must be one of {valid_levels}"
        )

    return levels[complexity_level]
