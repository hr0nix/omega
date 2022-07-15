import numpy as np
import jax

from omega.mcts.muzero import mcts, visualize_search_tree


class SimpleBandit:
    """
    Two arms.
    First arm gives reward 1 with probability 0.3 and -0.5 otherwise (expected value -0.05).
    Second arm gives reward 0.5 with probability 0.6 and -0.5 otherwise (expected value 0.1).
    Expected value under uniform policy is zero.
    Second arm should be preferred.
    """
    @property
    def initial_state(self):
        return np.asarray(0, dtype=np.int32)

    @property
    def absorbing_state(self):
        return np.asarray(1, dtype=np.int32)

    @property
    def absorbing_afterstate(self):
        return np.asarray(2, dtype=np.int32)

    @property
    def num_actions(self):
        return 2

    @property
    def num_outcomes(self):
        return 2

    def prediction(self, state):
        if state == self.initial_state:
            action_log_probs = np.log([0.5, 0.5])
            value = 0.5 * (0.3 * 1 + 0.7 * (-0.5)) + 0.5 * (0.6 * 0.5 + 0.4 * (-0.5))
        elif state.item() in [1, 2]:
            # Value of terminal states is zero
            action_log_probs, value = np.log([0.5, 0.5]), 0.0
        else:
            assert False, 'Unexpected state'
        return action_log_probs, np.asarray(value, dtype=np.int32)

    def afterstate_prediction(self, afterstate):
        if afterstate == 0:
            outcome_log_probs = np.log([0.3, 0.7])
            afterstate_value = 0.3 * 1 + 0.7 * (-0.5)
        elif afterstate == 1:
            outcome_log_probs = np.log([0.6, 0.4])
            afterstate_value = 0.6 * 0.5 + 0.4 * (-0.5)
        elif afterstate == self.absorbing_afterstate:
            outcome_log_probs = np.log([0.5, 0.5])
            afterstate_value = 0.0
        else:
            assert False, 'Unexpected afterstate'
        return outcome_log_probs, np.asarray(afterstate_value, dtype=np.float32)

    def dynamics(self, afterstate, outcome):
        assert outcome.shape == (2,)
        outcome = np.argmax(outcome)

        if afterstate == 0:
            if outcome == 0:
                reward = 1.0
            elif outcome == 1:
                reward = -0.5
            else:
                assert 'Unexpected outcome'
        elif afterstate == 1:
            if outcome == 0:
                reward = 0.5
            elif outcome == 1:
                reward = -0.5
            else:
                assert 'Unexpected outcome'
        elif afterstate == self.absorbing_afterstate:
            reward = 0.0
        else:
            assert 'Unexpected afterstate'

        next_state = 1
        return np.asarray(next_state, dtype=np.int32), np.asarray(reward, dtype=np.float32)

    def afterstate_dynamics(self, state, action):
        if action == 0:
            if state == self.initial_state:
                afterstate = 0  # First arm
            elif state == self.absorbing_state:
                afterstate = self.absorbing_afterstate
            else:
                assert False, 'Unexpected state'
        elif action == 1:
            if state == self.initial_state:
                afterstate = 1  # Second arm
            elif state == 1:
                afterstate = self.absorbing_afterstate
            else:
                assert False, 'Unexpected state'
        else:
            assert False, 'Unexpected action'

        return np.asarray(afterstate, dtype=np.int32)


def _test_bandit(
        bandit,
        expected_arm_values,
        search_policy='puct',
        result_policy='visit_count',
        num_simulations=60,
        search_tree_filename=None,
):
    with jax.disable_jit():
        policy_log_probs, value, search_tree, _ = mcts(
            initial_state=bandit.initial_state,
            rng=jax.random.PRNGKey(31337),
            prediction_fn=bandit.prediction,
            dynamics_fn=bandit.dynamics,
            afterstate_prediction_fn=bandit.afterstate_prediction,
            afterstate_dynamics_fn=bandit.afterstate_dynamics,
            num_actions=bandit.num_actions,
            num_chance_outcomes=bandit.num_outcomes,
            num_simulations=num_simulations,
            discount_factor=1.0,
            puct_c1=1.4,
            dirichlet_noise_alpha=0.35,
            root_exploration_fraction=0.25,
            search_policy=search_policy,
            result_policy=result_policy,
        )
        policy_probs = np.exp(policy_log_probs)
        assert np.argmax(policy_probs) == np.argmax(expected_arm_values)
        assert value >= search_tree['predicted_value'][0], 'There was no value improvement'

        if search_tree_filename is not None:
            visualize_search_tree(search_tree, search_tree_filename)


def test_simple_bandit_puct():
    _test_bandit(
        SimpleBandit(),
        expected_arm_values=[-0.05, 0.1],
        search_policy='puct',
        result_policy='visit_count',
        num_simulations=60,
        search_tree_filename='search_tree_puct.dot',
    )


def test_simple_bandit_pi_bar():
    _test_bandit(
        SimpleBandit(),
        expected_arm_values=[-0.05, 0.1],
        search_policy='pi_bar',
        result_policy='pi_bar',
        num_simulations=40,
        search_tree_filename='search_tree_pi_bar.dot',
    )
