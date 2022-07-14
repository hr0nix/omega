import math

import jax
import jax.tree_util
import jax.numpy as jnp

from omega.mcts.muzero import mcts


def number_of_first_child_visits():
    n_actions = 8
    max_action_prob = 0.5
    c_base = 2.0
    first_child_normalized_value = 0.4

    rest_action_prob = (1.0 - max_action_prob) / (n_actions - 1)

    def puct_cost(value, prob, parent_visits, child_visits):
        return value + prob * c_base * math.sqrt(parent_visits) / (child_visits + 1)

    max_simulations = 100
    for i in range(max_simulations):
        n = i + 1
        first_child_cost = puct_cost(first_child_normalized_value, max_action_prob, n + 1, n)
        other_children_cost = puct_cost(0.0, rest_action_prob, n + 1, 0)
        if other_children_cost > first_child_cost:
            print(f'First other child will be visited on simulation {i}')
            return

    print(f'Other children will not be visited within {max_simulations} simulations')


def mcts_test():
    discount_factor = 0.95
    action_probs = jnp.array([0., 0.9666665, 0., 0.03333333])
    reward_stddev = 0.06

    def value_func(r):
        return r / (1 - discount_factor)

    def reward_func(s):
        # We assume that any action from the state gives the same reward
        return jax.lax.select(s == 2, 2.0, 1.0)

    def prediction_fn(s, rng):
        return (
            jnp.log(action_probs), value_func(reward_func(s)) +
            jax.random.normal(rng) * reward_stddev / (1 - discount_factor)
        )

    def dynamics_fn(s, a, rng):
        next_state = jax.lax.select(s == 0, s + a + 1, s)  # Keep in the same state after root (which is state 0)
        return next_state, reward_func(next_state) + jax.random.normal(rng) * reward_stddev

    root_policy_log_probs, root_value = mcts(
        initial_state=jnp.zeros(shape=(), dtype=jnp.int32), rng=jax.random.PRNGKey(31337),
        prediction_fn=jax.tree_util.Partial(prediction_fn),
        dynamics_fn=jax.tree_util.Partial(dynamics_fn),
        num_actions=len(action_probs), num_simulations=30, discount_factor=discount_factor, puct_c1=1.25,
        dirichlet_noise_alpha=0.25, root_exploration_fraction=0.25,
    )
    print(f'Root policy probs: {jnp.exp(root_policy_log_probs)}')
    print(f'Root value: {root_value}')


if __name__ == '__main__':
    #number_of_first_child_visits()
    mcts_test()
