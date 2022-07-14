from omega.utils.collections import LinearPrioritizedSampler


def assert_sampling_probs(sampler, expected_item_to_prob, num_samples=10000, eps=1e-2):
    items = sampler.sample(num_items=num_samples)
    actual_item_to_freq = {}
    for item in items:
        actual_item_to_freq[item] = actual_item_to_freq.get(item, 0) + 1

    for item, expected_prob in expected_item_to_prob.items():
        actual_prob = actual_item_to_freq.get(item, 0) / num_samples
        assert abs(expected_prob - actual_prob) < eps


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
