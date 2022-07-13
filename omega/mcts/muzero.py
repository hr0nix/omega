import numpy as np

import jax.lax
import jax.numpy as jnp
import jax.experimental.host_callback

from ..utils import pytree


def get_global_child_index(tree, node_index, local_child_index):
    return tree['first_child_index'][node_index] + local_child_index


def get_global_parent_index(tree, node_index):
    return tree['parent_index'][node_index]


def get_local_child_index(tree, node_index):
    parent_index = get_global_parent_index(tree, node_index)
    return node_index - tree['first_child_index'][parent_index]


def get_state_value(tree, node_index):
    # Add eps to work around num_visits=0
    return tree['value_sum'][node_index] / (tree['num_visits'][node_index] + 1e-10)


def get_normalized_q_value(tree, node_index):
    # Can't assert this in JAX, but this function should only be applied to regular nodes
    # TODO: maybe return NaN if the node is chance?
    return jax.lax.cond(
        pred=tree['num_visits'][node_index] > 0,
        # Value of the chance node is the q-value
        true_fun=lambda _: get_normalized_value(tree, get_state_value(tree, node_index)),
        false_fun=lambda _: 0.0,
        operand=None)


def get_num_actions(tree):
    return tree['action_prior_probs'].shape[-1]


def get_num_chance_outcomes(tree):
    return tree['chance_outcome_prior_probs'].shape[-1]


def get_normalized_value(tree, value):
    # Add eps to work around min=max.
    # This is different from the original MuZero paper where they don't normalize if min=max,
    # but it shouldn't affect the behavior in any way.
    return (value - tree['min_max_value'][0]) / (tree['min_max_value'][1] - tree['min_max_value'][0] + 1e-10)


def update_value_normalization(tree, value):
    min_max_not_initialized = jnp.allclose(tree['min_max_value'], 0.0)
    tree = pytree.copy_structure(tree)
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
        child_index = get_global_child_index(tree, node_index, action)
        return tree['num_visits'][child_index]

    node_action_counts = jax.vmap(action_count, in_axes=(None, None, 0))(
        tree, node_index, jnp.arange(get_num_actions(tree), dtype=jnp.int32)
    )
    return jnp.log(node_action_counts) - jnp.log(jnp.sum(node_action_counts))


def get_policy_value(tree, node_index, policy_probs):
    def get_q_value(action):
        child_index = get_global_child_index(tree, node_index, action)
        return get_state_value(tree, child_index)
    q_values = jax.vmap(get_q_value)(jnp.arange(get_num_actions(tree), dtype=jnp.int32))
    return jnp.sum(q_values * policy_probs)


def init_node(
        tree, node_index, is_chance, state, reward, value,
        action_prior_log_probs, chance_outcome_prior_log_probs,
        rng, dirichlet_noise_alpha, exploration_fraction):
    rng, noise_key = jax.random.split(rng)

    action_prior_probs = jax.nn.softmax(action_prior_log_probs)
    # Add some dirichlet noise for exploration
    action_prior_noise = jax.random.dirichlet(rng, jnp.ones_like(action_prior_probs) * dirichlet_noise_alpha)
    action_prior_probs = (1.0 - exploration_fraction) * action_prior_probs + exploration_fraction * action_prior_noise

    chance_outcome_prior_probs = jax.nn.softmax(chance_outcome_prior_log_probs)

    tree = update_value_normalization(tree, value)
    tree['expanded'] = tree['expanded'].at[node_index].set(True)
    tree['is_chance'] = tree['is_chance'].at[node_index].set(is_chance)
    tree['predicted_value'] = tree['predicted_value'].at[node_index].set(value)
    tree['value_sum'] = tree['value_sum'].at[node_index].set(value)
    tree['num_visits'] = tree['num_visits'].at[node_index].set(1)
    tree['reward'] = tree['reward'].at[node_index].set(reward)
    tree['state'] = tree['state'].at[node_index].set(state)
    tree['chance_outcome_prior_probs'] = tree['chance_outcome_prior_probs'].at[node_index].set(
        chance_outcome_prior_probs)
    tree['action_prior_probs'] = tree['action_prior_probs'].at[node_index].set(action_prior_probs)

    first_child_index = tree['first_free_index'][0]
    max_num_children = max(get_num_actions(tree), get_num_chance_outcomes(tree))
    next_free_index = first_child_index + max_num_children
    tree['first_child_index'] = tree['first_child_index'].at[node_index].set(first_child_index)
    tree['parent_index'] = jax.lax.dynamic_update_slice(
        tree['parent_index'], jnp.full(max_num_children, node_index, dtype=jnp.int32), (first_child_index, ))
    tree['first_free_index'] = tree['first_free_index'].at[0].set(next_free_index)

    return tree


def make_tree(
        num_simulations, num_actions, num_chance_outcomes,
        initial_state, prediction_fn, rng,
        dirichlet_noise_alpha, root_exploration_fraction
):
    # Every expansion adds max(num_chance_outcomes, num_actions) nodes, plus there's the root node
    max_nodes_added_per_simulation = max(num_chance_outcomes, num_actions)
    tree_nodes_max_num = (num_simulations + 1) * max_nodes_added_per_simulation + 1

    tree = {
        'expanded': jnp.zeros(tree_nodes_max_num, dtype=jnp.bool_),
        'is_chance': jnp.zeros(tree_nodes_max_num, dtype=jnp.bool_),
        'first_child_index': jnp.zeros(tree_nodes_max_num, dtype=jnp.int32),
        'parent_index': jnp.zeros(tree_nodes_max_num, dtype=jnp.int32),
        'first_free_index': jnp.ones(1, dtype=jnp.int32),  # First free index is the next node after root
        'num_visits': jnp.zeros(tree_nodes_max_num, dtype=jnp.int32),
        'predicted_value': jnp.zeros(tree_nodes_max_num, dtype=jnp.float32),
        'value_sum': jnp.zeros(tree_nodes_max_num, dtype=jnp.float32),
        'reward': jnp.zeros(tree_nodes_max_num, dtype=jnp.float32),
        'action_prior_probs': jnp.zeros((tree_nodes_max_num, num_actions), dtype=jnp.float32),
        'chance_outcome_prior_probs': jnp.zeros((tree_nodes_max_num, num_chance_outcomes), dtype=jnp.float32),
        'state': jnp.zeros((tree_nodes_max_num,) + initial_state.shape, dtype=jnp.float32),
        'min_max_value': jnp.zeros(2, dtype=jnp.float32),
    }

    action_log_probs, root_value = prediction_fn(initial_state)

    return init_node(
        tree=tree, node_index=0, is_chance=False,
        state=initial_state, reward=0.0, value=root_value,
        action_prior_log_probs=action_log_probs,
        chance_outcome_prior_log_probs=jnp.zeros(num_chance_outcomes, dtype=jnp.float32),
        rng=rng,
        dirichlet_noise_alpha=dirichlet_noise_alpha, exploration_fraction=root_exploration_fraction
    )


# Implements the search policy from "Monte-Carlo tree search as regularized policy optimization",
# https://arxiv.org/abs/2007.12509
def get_pi_bar_policy(tree, node_index, c1):
    num_visits = tree['num_visits'][node_index]
    l = c1 * jnp.sqrt(num_visits) / num_visits
    pi = tree['action_prior_probs'][node_index]

    def get_normalized_action_q_value(action):
        child_index = get_global_child_index(tree, node_index, action)
        return get_normalized_q_value(tree, child_index)

    q_norm = jax.vmap(get_normalized_action_q_value)(jnp.arange(get_num_actions(tree), dtype=jnp.int32))

    a_min = jnp.max(q_norm + l * pi)
    a_max = jnp.max(q_norm + l)

    def get_pi_bar(a):
        return l * pi / (a - q_norm)

    def bin_search_condition(state):
        a_min, a_max = state
        return a_max - a_min > 0.01

    def bin_search_body(state):
        a_min, a_max = state
        a = 0.5 * (a_min + a_max)

        pi_bar = get_pi_bar(a)
        pi_bar_norm = jnp.sum(pi_bar)

        a_min, a_max = jax.lax.cond(
            pred=pi_bar_norm > 1,
            true_fun=lambda _: (a, a_max),
            false_fun=lambda _: (a_min, a),
            operand=None,
        )
        return a_min, a_max

    a_min, a_max = jax.lax.while_loop(cond_fun=bin_search_condition, body_fun=bin_search_body, init_val=(a_min, a_max))
    a = 0.5 * (a_min + a_max)
    pi_bar = get_pi_bar(a)
    pi_bar /= jnp.sum(pi_bar)  # Normalization found by binary search is imprecise

    return pi_bar

def pi_bar_search_policy(tree, node_index, c1):
    probs = get_pi_bar_policy(tree, node_index, c1)

    # TODO: the original paper proposes sampling, we use deterministic sampling here
    def child_score(action):
        child_index = get_global_child_index(tree, node_index, action)
        num_child_visits = tree['num_visits'][child_index]
        return probs[action] / (num_child_visits + 1)

    action_scores = jax.vmap(child_score)(jnp.arange(get_num_actions(tree), dtype=jnp.int32))
    return jnp.argmax(action_scores)


def puct_search_policy(tree, node_index, c1):
    def child_score(action):
        child_index = get_global_child_index(tree, node_index, action)
        num_parent_visits = tree['num_visits'][node_index]
        num_child_visits = tree['num_visits'][child_index]
        child_q_value = get_normalized_q_value(tree, child_index)
        parent_policy = tree['action_prior_probs'][node_index]
        c = c1 * jnp.sqrt(num_parent_visits) / (num_child_visits + 1)
        return child_q_value + parent_policy[action] * c

    action_scores = jax.vmap(child_score)(jnp.arange(get_num_actions(tree), dtype=jnp.int32))
    return jnp.argmax(action_scores)


def chance_outcome_search_policy(tree, node_index):
    def child_score(chance_outcome):
        child_index = get_global_child_index(tree, node_index, chance_outcome)
        num_child_visits = tree['num_visits'][child_index]
        parent_policy = tree['chance_outcome_prior_probs'][node_index]
        return parent_policy[chance_outcome] / (num_child_visits + 1)

    chance_outcome_scores = jax.vmap(child_score)(jnp.arange(get_num_chance_outcomes(tree), dtype=jnp.int32))
    return jnp.argmax(chance_outcome_scores)


def simulate(tree, puct_c1, search_policy):
    def while_condition(loop_state):
        current_node_index, _ = loop_state
        return tree['expanded'][current_node_index]

    def action_search_policy(current_index):
        if search_policy == 'pi_bar':
            return pi_bar_search_policy(tree, current_index, puct_c1)
        elif search_policy == 'puct':
            return puct_search_policy(tree, current_index, puct_c1)
        else:
            raise ValueError(f'Unknown search policy: {search_policy}')

    def loop_body(loop_state):
        current_index, depth = loop_state

        node_is_chance = tree['is_chance'][current_index]
        next_local_child_index = jax.lax.cond(
            pred=node_is_chance,
            true_fun=lambda _: chance_outcome_search_policy(tree, current_index),
            false_fun=lambda _: action_search_policy(current_index),
            operand=None,
        )
        child_index = get_global_child_index(tree, current_index, next_local_child_index)

        return child_index, depth + 1

    last_node_index, last_node_depth = jax.lax.while_loop(
        cond_fun=while_condition, body_fun=loop_body, init_val=(0, 0))

    return last_node_index, last_node_depth


def expand(tree, node_index, prediction_fn, afterstate_prediction_fn, dynamics_fn, afterstate_dynamics_fn, rng):
    rng, init_node_key = jax.random.split(rng, 2)

    parent_index = get_global_parent_index(tree, node_index)
    parent_state = tree['state'][parent_index]
    parent_is_chance = tree['is_chance'][parent_index]
    local_child_index = get_local_child_index(tree, node_index)

    # We use a sequence of while loops to work around the cond-inside-vmap issue,
    # which causes both branches of cond to evaluate, wasting precious FLOPs
    # https://github.com/google/jax/issues/2947#issuecomment-623745670

    child_state = jnp.zeros_like(parent_state)
    reward = 0.0
    value = 0.0
    action_log_probs = jnp.zeros(get_num_actions(tree), dtype=jnp.float32)
    chance_outcome_log_probs = jnp.zeros(get_num_chance_outcomes(tree), dtype=jnp.float32)

    def chance_node_expansion_cond(loop_state):
        is_first_iter = loop_state[-1]
        return jnp.logical_and(parent_is_chance, is_first_iter)
    def chance_node_expansion(_):
        num_chance_outcomes = get_num_chance_outcomes(tree)
        chance_outcome_one_hot = jax.nn.one_hot(local_child_index, num_classes=num_chance_outcomes, dtype=jnp.float32)
        child_state, reward = dynamics_fn(parent_state, chance_outcome_one_hot)
        action_log_probs, value = prediction_fn(child_state)
        return child_state, reward, value, action_log_probs, False
    child_state, reward, value, action_log_probs, _ = jax.lax.while_loop(
        cond_fun=chance_node_expansion_cond, body_fun=chance_node_expansion,
        init_val=(child_state, reward, value, action_log_probs, True)
    )

    def regular_node_expansion_cond(loop_state):
        is_first_iter = loop_state[-1]
        return jnp.logical_and(jnp.logical_not(parent_is_chance), is_first_iter)
    def regular_node_expansion(_):
        afterstate = afterstate_dynamics_fn(parent_state, local_child_index)
        chance_outcome_log_probs, value = afterstate_prediction_fn(afterstate)
        return afterstate, value, chance_outcome_log_probs, False
    child_state, value, chance_outcome_log_probs, _ = jax.lax.while_loop(
        cond_fun=regular_node_expansion_cond, body_fun=regular_node_expansion,
        init_val=(child_state, value, chance_outcome_log_probs, True)
    )

    return init_node(
        tree, node_index, is_chance=jnp.logical_not(parent_is_chance), state=child_state, value=value, reward=reward,
        action_prior_log_probs=action_log_probs, chance_outcome_prior_log_probs=chance_outcome_log_probs,
        rng=init_node_key,
        dirichlet_noise_alpha=1.0, exploration_fraction=0.0,  # No exploration in nodes other than root
    )


def backprop(tree, leaf_index, discount_factor):
    def while_condition(loop_state):
        tree, current_node_index, value = loop_state
        return current_node_index != 0

    def loop_body(loop_state):
        tree, current_index, value = loop_state

        parent_index = get_global_parent_index(tree, current_index)

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
        prediction_fn, afterstate_prediction_fn,
        dynamics_fn, afterstate_dynamics_fn,
        num_actions, num_chance_outcomes, num_simulations,
        discount_factor, puct_c1,
        dirichlet_noise_alpha, root_exploration_fraction,
        search_policy='puct',
        result_policy='visit_count'
):
    make_tree_key, simulation_loop_key = jax.random.split(rng)

    tree = make_tree(
        num_simulations, num_actions, num_chance_outcomes,
        initial_state, prediction_fn, make_tree_key,
        dirichlet_noise_alpha, root_exploration_fraction)

    def simulation_iteration(simulation_index, loop_state):
        rng, tree, max_depth = loop_state
        rng, expansion_key = jax.random.split(rng)

        leaf_index, leaf_depth = simulate(tree, puct_c1, search_policy)
        expanded_tree = expand(
            tree, leaf_index,
            prediction_fn, afterstate_prediction_fn, dynamics_fn, afterstate_dynamics_fn,
            expansion_key)
        backproped_tree = backprop(expanded_tree, leaf_index, discount_factor)

        return rng, backproped_tree, jnp.maximum(max_depth, leaf_depth)

    _, updated_tree, max_depth = jax.lax.fori_loop(
        lower=0, upper=num_simulations, body_fun=simulation_iteration,
        init_val=(simulation_loop_key, tree, 0)
    )

    if result_policy == 'visit_count':
        root_mcts_policy_log_probs = get_visitation_based_policy(tree=updated_tree, node_index=0)
        # Accumulated value at the root is exactly the visitation count policy value
        root_mcts_value = get_state_value(tree=updated_tree, node_index=0)
    elif result_policy == 'pi_bar':
        pi_bar_policy = get_pi_bar_policy(tree=updated_tree, node_index=0, c1=puct_c1)
        root_mcts_policy_log_probs = jnp.log(pi_bar_policy)
        root_mcts_value = get_policy_value(tree=updated_tree, node_index=0, policy_probs=pi_bar_policy)
    else:
        raise ValueError(f'Unknown result policy: {result_policy}')

    stats = {
        'mcts_search_depth': max_depth,
    }

    return root_mcts_policy_log_probs, root_mcts_value, updated_tree, stats


def visualize_search_tree(tree, filename):
    import pygraphviz
    graph = pygraphviz.AGraph(directed=True)

    def format_policy(probs):
        return '[' + ' '.join(f'{prob:.2f}' for prob in probs) + ']'

    def as_scalar(device_array):
        return np.asarray(device_array).item()

    def node_label(node_index):
        label = ''
        if tree['expanded'][node_index]:
            value = get_state_value(tree, node_index)
            label += f'N: {as_scalar(tree["num_visits"][node_index])}\n'
            label += f'V: {as_scalar(value):.4f}\n'
            label += f'PV: {as_scalar(tree["predicted_value"][node_index]):.4f}\n'
            label += f'NV: {as_scalar(get_normalized_value(tree, value)):.4f}\n'

            if tree['is_chance'][node_index]:
                label += f'CP: {format_policy(tree["chance_outcome_prior_probs"][node_index])}\n'
            else:
                label += f'R: {as_scalar(tree["reward"][node_index]):.3f}\n'
                label += f'P: {format_policy(tree["action_prior_probs"][node_index])}\n'
        return label

    def add_node(node_index):
        label = node_label(node_index)
        color = 'cyan' if tree['is_chance'][node_index] else 'green' if tree['expanded'][node_index] else 'red'
        graph.add_node(node_index, label=label, fillcolor=color, shape='box', style='filled')

    add_node(0)
    for child_index in range(1, np.asarray(tree['first_free_index']).item()):
        add_node(child_index)
        parent_index = get_global_parent_index(tree, child_index)
        action_or_chance_outcome = get_local_child_index(tree, child_index)
        graph.add_edge(parent_index, child_index, label=action_or_chance_outcome)

    graph.draw(filename, prog='dot')
