import rlax
import pytest


def test_value_reward_representation():
    value = -0.01
    min_value = -1.0
    max_value = 1.0
    bins = 32
    value_2_hot = rlax.transform_to_2hot(value, min_value, max_value, bins)
    reconstructed_value = rlax.transform_from_2hot(value_2_hot, min_value, max_value, bins)
    assert reconstructed_value == pytest.approx(value, rel=1e-3)


def test_value_reward_representation_clamped():
    value = -2.0
    min_value = -1.0
    max_value = 1.0
    bins = 32
    value_2_hot = rlax.transform_to_2hot(value, min_value, max_value, bins)
    reconstructed_value = rlax.transform_from_2hot(value_2_hot, min_value, max_value, bins)
    assert reconstructed_value == pytest.approx(min_value, rel=1e-3)
