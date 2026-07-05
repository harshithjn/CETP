"""Unit tests for cetp.workload_config."""

import pytest

from cetp.workload_config import (
    VALID_MODELS,
    WORKLOAD_COMPLEXITY_MAP,
    get_workload_params,
)


class TestGetWorkloadParams:
    @pytest.mark.parametrize(
        "model_name,complexity_level",
        [
            (model, level)
            for model in WORKLOAD_COMPLEXITY_MAP
            for level in WORKLOAD_COMPLEXITY_MAP[model]
        ],
    )
    def test_valid_model_and_level_returns_correct_tuple(self, model_name, complexity_level):
        """Every valid (model, complexity_level) combination returns its exact tuple."""
        expected = WORKLOAD_COMPLEXITY_MAP[model_name][complexity_level]
        assert get_workload_params(model_name, complexity_level) == expected

    def test_all_20_combinations_covered(self):
        """Sanity check that the parametrized test above covers all 20 combinations."""
        total = sum(len(levels) for levels in WORKLOAD_COMPLEXITY_MAP.values())
        assert total == 20

    def test_invalid_model_raises_value_error(self):
        """An unknown model name raises ValueError listing the valid models."""
        with pytest.raises(ValueError) as exc_info:
            get_workload_params("not_a_real_model", 3)
        message = str(exc_info.value)
        assert "not_a_real_model" in message
        for model in VALID_MODELS:
            assert model in message

    def test_complexity_level_zero_raises_value_error(self):
        """complexity_level=0 is out of range and raises ValueError listing valid levels."""
        with pytest.raises(ValueError) as exc_info:
            get_workload_params("resnet18", 0)
        message = str(exc_info.value)
        assert "0" in message
        assert "[1, 2, 3, 4, 5]" in message

    def test_complexity_level_six_raises_value_error(self):
        """complexity_level=6 is out of range and raises ValueError listing valid levels."""
        with pytest.raises(ValueError) as exc_info:
            get_workload_params("resnet50", 6)
        message = str(exc_info.value)
        assert "6" in message
        assert "[1, 2, 3, 4, 5]" in message

    def test_complexity_level_negative_raises_value_error(self):
        """complexity_level=-1 is out of range and raises ValueError listing valid levels."""
        with pytest.raises(ValueError) as exc_info:
            get_workload_params("mobilenet", -1)
        message = str(exc_info.value)
        assert "-1" in message
        assert "[1, 2, 3, 4, 5]" in message


class TestConfigShape:
    def test_exactly_four_models(self):
        """WORKLOAD_COMPLEXITY_MAP has exactly 4 models."""
        assert len(WORKLOAD_COMPLEXITY_MAP) == 4

    def test_valid_models_matches_map_keys(self):
        """VALID_MODELS lists exactly the keys of WORKLOAD_COMPLEXITY_MAP."""
        assert set(VALID_MODELS) == set(WORKLOAD_COMPLEXITY_MAP.keys())

    @pytest.mark.parametrize("model_name", ["resnet18", "resnet50", "mobilenet", "distilbert"])
    def test_each_model_has_exactly_five_levels(self, model_name):
        """Each model in WORKLOAD_COMPLEXITY_MAP has exactly 5 complexity levels."""
        assert len(WORKLOAD_COMPLEXITY_MAP[model_name]) == 5
        assert set(WORKLOAD_COMPLEXITY_MAP[model_name].keys()) == {1, 2, 3, 4, 5}
