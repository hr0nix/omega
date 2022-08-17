import jax.numpy as jnp

from omega.agents.nethack_muzero_agent import (
    compute_training_targets, make_reward_transform_pair, compute_one_hot_targets
)


def test_compute_training_targets():
    trajectory = {
        'actions': jnp.asarray([0, 1, 0, 1], dtype=jnp.int32),
        'rewards': jnp.asarray([0.1, 0.2, 0.3, 0.4], dtype=jnp.float32),
        'done': jnp.asarray([False, False, True, False], dtype=jnp.bool_),
        'mcts_reanalyze': {
            'state_values': jnp.asarray([4.0, 3.0, 2.0, 1.0], dtype=jnp.float32),
            'log_action_probs': jnp.log(jnp.asarray([
                [0.3, 0.7], [0.1, 0.9], [0.2, 0.8], [0.4, 0.6],
            ])),
        }
    }
    df = 0.95
    targets = compute_training_targets(trajectory, num_unroll_steps=3, discount_factor=df * df)
    assert jnp.allclose(
        targets['state_value_scalar'],
        jnp.asarray([
            [4.0, 3.0, 2.0],
            [3.0, 2.0, 0.0],
            [2.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
        ], dtype=jnp.float32)
    )
    assert jnp.allclose(
        targets['afterstate_value_scalar'],
        jnp.asarray([
            [0.1 + df * 3.0, 0.2 + df * 2.0, 0.3 + df * 0.0],
            [0.2 + df * 2.0, 0.3 + df * 0.0, 0.0 + df * 0.0],
            [0.3 + df * 0.0, 0.0 + df * 0.0, 0.0 + df * 0.0],
            [0.4 + df * 0.0, 0.0 + df * 0.0, 0.0 + df * 0.0],
        ], dtype=jnp.float32)
    )
    assert jnp.allclose(
        targets['reward_scalar'],
        jnp.asarray([
            [0.1, 0.2, 0.3],
            [0.2, 0.3, 0.0],
            [0.3, 0.0, 0.0],
            [0.4, 0.0, 0.0],
        ], dtype=jnp.float32)
    )
    assert jnp.array_equal(
        targets['action'],
        jnp.asarray([
            [0, 1, 0],
            [1, 0, 1],
            [0, 1, 0],
            [1, 0, 0],
        ], dtype=jnp.int32)
    )
    assert jnp.array_equal(
        targets['next_state_is_terminal_or_after'],
        jnp.asarray([
            [False, False, True],
            [False, True, True],
            [True, True, True],
            [False, False, False],
        ], dtype=jnp.bool_)
    )
    assert jnp.allclose(
        targets['policy'],
        jnp.asarray([
            [[0.3, 0.7], [0.1, 0.9], [0.2, 0.8]],
            [[0.1, 0.9], [0.2, 0.8], [0.5, 0.5]],
            [[0.2, 0.8], [0.5, 0.5], [0.5, 0.5]],
            [[0.4, 0.6], [0.5, 0.5], [0.5, 0.5]],
        ], dtype=jnp.float32)
    )


def test_convert_to_discrete_representation():
    targets_scalar = {
        'reward_scalar': jnp.asarray([-1.1, 0.0, 0.8, 1.0], dtype=jnp.float32),
    }
    transform_pair = make_reward_transform_pair(min_value=-1.0, max_value=1.0, num_bins=3)
    targets_discrete = compute_one_hot_targets(targets_scalar, transform_pair.apply)
    assert jnp.allclose(
        targets_discrete['reward_discrete'],
        jnp.asarray([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.2, 0.8], [0.0, 0.0, 1.0]], dtype=jnp.float32),
        rtol=1e-4,
    )
