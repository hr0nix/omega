import pytest

from omega.utils.collections import LinearPrioritizedSampler


def assert_sampling_probs(sampler, expected_item_to_prob, num_samples=10000, eps=1e-2):
    items, weights = sampler.sample(num_items=num_samples)
    actual_item_to_freq = {}
    actual_item_to_weight = {}
    for i in range(len(items)):
        actual_item_to_freq[items[i]] = actual_item_to_freq.get(items[i], 0) + 1
        actual_item_to_weight[items[i]] = actual_item_to_weight.get(items[i], 0) + weights[i]

    for item, expected_prob in expected_item_to_prob.items():
        actual_prob = actual_item_to_freq.get(item, 0) / num_samples
        actual_weight = actual_item_to_weight.get(item, 0)
        assert actual_prob == pytest.approx(expected_prob, abs=eps)
        # Importance weights should be equally balanced between all items
        assert actual_weight / num_samples == pytest.approx(1.0 / len(expected_item_to_prob), abs=eps)


def test_linear_sampler_add():
    sampler = LinearPrioritizedSampler(max_items=3)

    sampler.add('1', 1.0)
    assert_sampling_probs(sampler, {'1': 1.0})

    sampler.add('2', 2.0)
    assert_sampling_probs(sampler, {'1': 1.0 / 3.0, '2': 2.0 / 3.0})

    sampler.add('3', 3.0)
    assert_sampling_probs(sampler, {'1': 1.0 / 6.0, '2': 2.0 / 6.0, '3': 3.0 / 6.0})


def test_linear_sampler_remove():
    sampler = LinearPrioritizedSampler(max_items=3)

    sampler.add('1', 1.0)
    sampler.add('2', 2.0)
    sampler.add('3', 3.0)
    assert_sampling_probs(sampler, {'1': 1.0 / 6.0, '2': 2.0 / 6.0, '3': 3.0 / 6.0})

    sampler.remove('1')
    assert_sampling_probs(sampler, {'2': 2.0 / 5.0, '3': 3.0 / 5.0})

    sampler.remove('2')
    assert_sampling_probs(sampler, {'3': 1.0})


def test_linear_sampler_remove_last():
    sampler = LinearPrioritizedSampler(max_items=3)

    sampler.add('1', 1.0)
    sampler.add('2', 2.0)
    sampler.add('3', 3.0)
    assert_sampling_probs(sampler, {'1': 1.0 / 6.0, '2': 2.0 / 6.0, '3': 3.0 / 6.0})

    sampler.remove('3')
    assert_sampling_probs(sampler, {'1': 1.0 / 3.0, '2': 2.0 / 3.0})

    sampler.remove('2')
    assert_sampling_probs(sampler, {'1': 1.0})


def test_linear_sampler_update():
    sampler = LinearPrioritizedSampler(max_items=3)

    sampler.add('1', 1.0)
    sampler.add('2', 2.0)
    sampler.add('3', 3.0)
    assert_sampling_probs(sampler, {'1': 1.0 / 6.0, '2': 2.0 / 6.0, '3': 3.0 / 6.0})

    sampler.update_priority('2', 4.0)
    assert_sampling_probs(sampler, {'1': 1.0 / 8.0, '2': 4.0 / 8.0, '3': 3.0 / 8.0})

    sampler.update_priority('1', 2.0)
    assert_sampling_probs(sampler, {'1': 2.0 / 9.0, '2': 4.0 / 9.0, '3': 3.0 / 9.0})
