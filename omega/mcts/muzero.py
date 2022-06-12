import jax.lax
import jax.numpy as jnp
import jax.experimental.host_callback

from ..utils import pytree


def get_child_index(tree, node_index, action_index):
    return tree['first_child_index'][node_index] + action_index


def get_parent_index(tree, node_index):
    return tree['parent_index'][node_index]


def get_incoming_action(tree, node_index):
    parent_index = get_parent_index(tree, node_index)
    return node_index - tree['first_child_index'][parent_index]


def get_num_actions(tree):
    return tree['prior_policy_probs'].shape[-1]


def get_state_value(tree, node_index):
    # Add eps to work around num_visits=0
    return tree['value_sum'][node_index] / (tree['num_visits'][node_index] + 1e-10)


def get_normalized_value(tree, value):
    # Add eps to work around min=max.
    # This is different from the original MuZero paper where they don't normalize if min=max,
    # but it shouldn't affect the behavior in any way.
    return (value - tree['min_max_value'][0]) / (tree['min_max_value'][1] - tree['min_max_value'][0] + 1e-10)


def update_value_normalization(tree, value):
    min_max_not_initialized = jnp.allclose(tree['min_max_value'], 0.0)
    tree = pytree.copy(tree)
    tree['min_max_value'] = jax.lax.cond(
        pred=min_max_not_initialized,
        true_fun=lambda _: jnp.ones(2, dtype=jnp.float32) * value,
        false_fun=lambda _: jnp.array(
            [
                jnp.minimum(tree['min_max_value'][0], value),
                jnp.maximum(tree['min_max_value'][1], value),
            ],
            dtype=jnp.float32,
        ),
        operand=None,
    )
    return tree


def get_visitation_based_policy(tree, node_index):
    def action_count(tree, node_index, action):
        child_index = get_child_index(tree, node_index, action)
        return tree['num_visits'][child_index]

    node_action_counts = jax.vmap(action_count, in_axes=(None, None, 0))(
        tree, node_index, jnp.arange(get_num_actions(tree), dtype=jnp.int32)
    )
    return jnp.log(node_action_counts) - jnp.log(jnp.sum(node_action_counts))


def init_node(
        tree, node_index, state, reward, prediction_fn, rng,
        dirichlet_noise_alpha, exploration_fraction):
    prediction_key, noise_key = jax.random.split(rng)

    prior_policy_log_probs, value = prediction_fn(state, prediction_key)
    prior_policy_probs = jax.nn.softmax(prior_policy_log_probs)
    num_actions = prior_policy_log_probs.shape[-1]

    # Add some dirichlet noise for exploration
    prior_policy_noise = jax.random.dirichlet(noise_key, jnp.ones(num_actions) * dirichlet_noise_alpha)
    prior_policy_probs = (1.0 - exploration_fraction) * prior_policy_probs + exploration_fraction * prior_policy_noise

    tree = update_value_normalization(tree, value)
    tree['expanded'] = tree['expanded'].at[node_index].set(True)
    tree['predicted_value'] = tree['predicted_value'].at[node_index].set(value)
    tree['value_sum'] = tree['value_sum'].at[node_index].set(value)
    tree['num_visits'] = tree['num_visits'].at[node_index].set(1)
    tree['reward'] = tree['reward'].at[node_index].set(reward)
    tree['prior_policy_probs'] = tree['prior_policy_probs'].at[node_index].set(prior_policy_probs)
    tree['state'] = tree['state'].at[node_index].set(state)

    first_child_index = tree['first_free_index'][0]
    next_free_index = first_child_index + num_actions
    tree['first_child_index'] = tree['first_child_index'].at[node_index].set(first_child_index)
    tree['parent_index'] = jax.lax.dynamic_update_slice(
        tree['parent_index'], jnp.ones(num_actions, dtype=jnp.int32) * node_index, (first_child_index, ))
    tree['first_free_index'] = tree['first_free_index'].at[0].set(next_free_index)

    return tree


def make_tree(
        num_simulations, num_actions, initial_state, prediction_fn, rng,
        dirichlet_noise_alpha, root_exploration_fraction
):
    tree_nodes_max_num = (num_simulations + 1) * num_actions  # Every expansion adds num_actions nodes to the tree

    tree = {
        'expanded': jnp.zeros(tree_nodes_max_num, dtype=jnp.bool_),
        'first_child_index': jnp.zeros(tree_nodes_max_num, dtype=jnp.int32),
        'parent_index': jnp.zeros(tree_nodes_max_num, dtype=jnp.int32),
        'first_free_index': jnp.ones(1, dtype=jnp.int32),  # First free index is the next node after root
        'num_visits': jnp.zeros(tree_nodes_max_num, dtype=jnp.int32),
        'predicted_value': jnp.zeros(tree_nodes_max_num, dtype=jnp.float32),
        'value_sum': jnp.zeros(tree_nodes_max_num, dtype=jnp.float32),
        'reward': jnp.zeros(tree_nodes_max_num, dtype=jnp.float32),
        'prior_policy_probs': jnp.zeros((tree_nodes_max_num, num_actions), dtype=jnp.float32),
        'state': jnp.zeros((tree_nodes_max_num,) + initial_state.shape, dtype=jnp.float32),
        'min_max_value': jnp.zeros(2, dtype=jnp.float32),
    }

    return init_node(
        tree=tree, node_index=0,
        state=initial_state, reward=0.0,
        prediction_fn=prediction_fn,
        rng=rng,
        dirichlet_noise_alpha=dirichlet_noise_alpha, exploration_fraction=root_exploration_fraction
    )


def puct_search_policy(tree, node_index, discount_factor, c1):
    def child_score(tree, parent_index, action):
        child_index = get_child_index(tree, parent_index, action)
        num_parent_visits = tree['num_visits'][parent_index]
        num_child_visits = tree['num_visits'][child_index]

        child_value = jax.lax.cond(
            pred=num_child_visits > 0,
            true_fun=lambda _: get_normalized_value(
                tree,
                tree['reward'][child_index] + discount_factor * get_state_value(tree, child_index)
            ),
            false_fun=lambda _: 0.0,
            operand=None)

        parent_policy = tree['prior_policy_probs'][parent_index]

        c = c1 * jnp.sqrt(num_parent_visits) / (num_child_visits + 1)
        return child_value + parent_policy[action] * c

    action_scores = jax.vmap(child_score, in_axes=(None, None, 0), out_axes=0)(
        tree, node_index, jnp.arange(get_num_actions(tree), dtype=jnp.int32),
    )
    return jnp.argmax(action_scores)


def simulate(tree, discount_factor, puct_c1):
    def while_condition(loop_state):
        tree, current_node_index, _ = loop_state
        return tree['expanded'][current_node_index]

    def loop_body(loop_state):
        tree, current_index, depth = loop_state

        action = puct_search_policy(tree, current_index, discount_factor, puct_c1)
        child_index = get_child_index(tree, current_index, action)

        return tree, child_index, depth + 1

    _, last_node_index, last_node_depth = jax.lax.while_loop(
        cond_fun=while_condition, body_fun=loop_body, init_val=(tree, 0, 0))

    return last_node_index, last_node_depth


def expand(tree, node_index, prediction_fn, dynamics_fn, rng):
    prediction_key, dynamics_key = jax.random.split(rng)

    parent_index = get_parent_index(tree, node_index)
    parent_state = tree['state'][parent_index]
    action = get_incoming_action(tree, node_index)
    child_state, reward = dynamics_fn(parent_state, action, dynamics_key)

    tree = init_node(
        tree, node_index, child_state, reward, prediction_fn, prediction_key,
        dirichlet_noise_alpha=1.0, exploration_fraction=0.0,  # No exploration in nodes other than root
    )

    return tree


def backprop(tree, leaf_index, discount_factor):
    def while_condition(loop_state):
        tree, current_node_index, value = loop_state
        return current_node_index != 0

    def loop_body(loop_state):
        tree, current_index, value = loop_state

        parent_index = get_parent_index(tree, current_index)

        value = value * discount_factor + tree['reward'][current_index]
        tree = update_value_normalization(tree, value)
        tree['value_sum'] = tree['value_sum'].at[parent_index].add(value)
        tree['num_visits'] = tree['num_visits'].at[parent_index].add(1)

        return tree, parent_index, value

    leaf_value = tree['value_sum'][leaf_index]  # Filled by init_node together with num_visits
    tree, root_node_index, root_value = jax.lax.while_loop(
        cond_fun=while_condition, body_fun=loop_body, init_val=(tree, leaf_index, leaf_value))

    return tree


def mcts(
        initial_state, rng,
        prediction_fn, dynamics_fn,
        num_actions, num_simulations,
        discount_factor, puct_c1,
        dirichlet_noise_alpha, root_exploration_fraction,
):
    make_tree_key, simulation_loop_key = jax.random.split(rng)

    tree = make_tree(
        num_simulations, num_actions, initial_state, prediction_fn, make_tree_key,
        dirichlet_noise_alpha, root_exploration_fraction)

    def simulation_iteration(simulation_index, loop_state):
        rng, tree, max_depth = loop_state
        rng, expansion_key = jax.random.split(rng)

        leaf_index, leaf_depth = simulate(tree, discount_factor, puct_c1)
        expanded_tree = expand(tree, leaf_index, prediction_fn, dynamics_fn, expansion_key)
        backproped_tree = backprop(expanded_tree, leaf_index, discount_factor)

        return rng, backproped_tree, jnp.maximum(max_depth, leaf_depth)

    _, updated_tree, max_depth = jax.lax.fori_loop(
        lower=0, upper=num_simulations, body_fun=simulation_iteration,
        init_val=(simulation_loop_key, tree, 0)
    )

    root_mcts_policy_log_probs = get_visitation_based_policy(tree=updated_tree, node_index=0)
    root_mcts_value = get_state_value(tree=updated_tree, node_index=0)

    stats = {
        'mcts_search_depth': max_depth,
    }

    return root_mcts_policy_log_probs, root_mcts_value, stats

