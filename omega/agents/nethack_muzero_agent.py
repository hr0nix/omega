import functools

from functools import partial
from typing import Callable

from absl import logging

import chex
import flax.struct
import flax.training.train_state
import flax.training.checkpoints
import jax.numpy as jnp
import jax.random
import jax.tree_util
import jax.experimental.host_callback
import optax
import rlax

from ..utils import pytree
from ..math import entropy, discretize_onehot, undiscretize_expected
from ..utils.flax import merge_params
from ..utils.profiling import timeit
from ..models.nethack_muzero import NethackPerceiverMuZeroModel
from ..mcts.muzero import mcts
from .trainable_agent import JaxTrainableAgentBase


class NethackMuZeroAgent(JaxTrainableAgentBase):
    CONFIG = flax.core.frozen_dict.FrozenDict({
        'lr': 1e-3,
        'discount_factor': 0.99,
        'model_config': {},
        'num_mcts_simulations': 30,
        'mcts_puct_c1': 1.25,
        'mcts_dirichlet_noise_alpha': 0.2,
        'mcts_root_exploration_fraction': 0.2,
        'num_train_unroll_steps': 5,
        'reanalyze_batch_size': 8,
        'policy_loss_weight': 1.0,
        'value_loss_weight': 1.0,
        'reward_loss_weight': 1.0,
        'state_similarity_loss_weight': 1.0,
        'reward_lookup': {
            -0.01: 0,
            0.0: 1,
            1.0: 2,
        }
    })

    class TrainState(flax.training.train_state.TrainState):
        representation_fn: Callable = flax.struct.field(pytree_node=False)
        dynamics_fn: Callable = flax.struct.field(pytree_node=False)
        prediction_fn: Callable = flax.struct.field(pytree_node=False)
        step_index: int = 0

    def __init__(self, *args, model_factory=NethackPerceiverMuZeroModel, replay_buffer, config=None, **kwargs):
        super(NethackMuZeroAgent, self).__init__(*args, **kwargs)

        self._config = self.CONFIG.copy(config or {})
        # TODO: MuZero uses prioritized replay
        self._replay_buffer = replay_buffer
        self._build_model(model_factory)

    def _build_model(self, model_factory):
        self._model = model_factory(
            num_actions=self.action_space.n, **self._config['model_config'],
            reward_dim=len(self._config['reward_lookup']),
            name='mu_zero_model')

        representation_params = self._model.init(
            self.next_random_key(), *self._make_fake_representation_inputs(),
            method=self._model.representation,
            deterministic=False, rng=self.next_random_key())
        dynamics_params = self._model.init(
            self.next_random_key(), *self._make_fake_dynamics_inputs(),
            method=self._model.dynamics,
            deterministic=False, rng=self.next_random_key())
        prediction_params = self._model.init(
            self.next_random_key(), *self._make_fake_prediction_inputs(),
            method=self._model.prediction,
            deterministic=False, rng=self.next_random_key())

        # Merge params from different initializations, some values will be overridden
        model_params = merge_params(representation_params, dynamics_params, prediction_params)

        optimizer = optax.adam(learning_rate=self._config['lr'])

        self._train_state = self.TrainState.create(
            params=model_params, tx=optimizer,
            apply_fn=self._model.apply,
            representation_fn=functools.partial(
                self._model.apply, method=self._model.representation, deterministic=False),
            dynamics_fn=functools.partial(
                self._model.apply, method=self._model.dynamics, deterministic=False),
            prediction_fn=functools.partial(
                self._model.apply, method=self._model.prediction, deterministic=False),
        )

    def _make_fake_observation_trajectory(self):
        return {
            key: jnp.zeros(shape=(1, ) + desc.shape, dtype=desc.dtype)
            for key, desc in self.observation_space.spaces.items()
        }

    def _make_fake_latent_state(self):
        latent_state_shape = self._model.apply({}, method=self._model.latent_state_shape)
        return jnp.zeros(shape=latent_state_shape, dtype=jnp.float32)

    def _make_fake_action(self):
        return jnp.zeros(shape=(), dtype=jnp.int32)

    def _make_fake_representation_inputs(self):
        return self._make_fake_observation_trajectory(),

    def _make_fake_dynamics_inputs(self):
        return self._make_fake_latent_state(), self._make_fake_action()

    def _make_fake_prediction_inputs(self):
        return self._make_fake_latent_state(),

    def try_load_from_checkpoint(self, path):
        checkpoint_path = flax.training.checkpoints.latest_checkpoint(path)
        if checkpoint_path is None:
            logging.warning('No checkpoints available at {}'.format(path))
            return 0
        else:
            logging.info('State will be loaded from checkpoint {}'.format(checkpoint_path))
            self._train_state = flax.training.checkpoints.restore_checkpoint(checkpoint_path, self._train_state)
            return self._train_state.step_index + 1

    def save_to_checkpoint(self, checkpoint_path, step):
        self._train_state = self._train_state.replace(step_index=step)
        flax.training.checkpoints.save_checkpoint(
            checkpoint_path, self._train_state, step=step, keep=1, overwrite=True)

    def train_on_batch(self, trajectory_batch):
        if not self._replay_buffer.empty:
            training_batch, reanalyse_stats = self._make_next_training_batch(trajectory_batch)
            training_stats = self._train(training_batch)
            stats = pytree.update(reanalyse_stats, training_stats)
        else:
            # Skip the first training step while the replay buffer is still empty
            stats = {}

        # Add training data after reanalyze so that we don't try to reanalyze most recent  trajectories
        self._replay_buffer.add_trajectory_batch(trajectory_batch)

        return stats

    def act_on_batch(self, observation_batch):
        return self._act_on_batch_jit(observation_batch, self._train_state, self.next_random_key())

    def _compute_mcts_statistics(self, observation_trajectory_batch, train_state, rng):
        representation_key, mcts_key = jax.random.split(rng)

        batch_size, num_timestamps = jax.tree_leaves(observation_trajectory_batch)[0].shape[:2]

        represent_observation_trajectory_batch = jax.vmap(train_state.representation_fn, in_axes=(None, 0, 0))
        represenation_key_batch = jax.random.split(representation_key, batch_size)
        latent_state_trajectory_batch = represent_observation_trajectory_batch(
            train_state.params, observation_trajectory_batch, represenation_key_batch)

        def dynamics_fn(params, state, action, rng):
            next_state, reward_logits = train_state.dynamics_fn(params, state, action, rng)
            # Convert reward back to continuous form
            return next_state, undiscretize_expected(reward_logits, self._config['reward_lookup'])

        mcts_func = functools.partial(
            mcts,
            num_actions=self.action_space.n,
            prediction_fn=jax.tree_util.Partial(train_state.prediction_fn, train_state.params),
            dynamics_fn=jax.tree_util.Partial(dynamics_fn, train_state.params),
            discount_factor=self._config['discount_factor'],
            num_simulations=self._config['num_mcts_simulations'],
            puct_c1=self._config['mcts_puct_c1'],
            dirichlet_noise_alpha=self._config['mcts_dirichlet_noise_alpha'],
            root_exploration_fraction=self._config['mcts_root_exploration_fraction'],
        )
        trajectory_mcts = jax.vmap(mcts_func, in_axes=(0, 0), out_axes=(0, 0, 0))
        trajectory_batch_mcts = jax.vmap(trajectory_mcts, in_axes=(0, 0), out_axes=(0, 0, 0))
        mcts_key_batch = jax.random.split(mcts_key, batch_size * num_timestamps).reshape(batch_size, num_timestamps, 2)
        mcts_policy_log_probs, mcts_value, mcts_stats = trajectory_batch_mcts(
            latent_state_trajectory_batch, mcts_key_batch)

        return mcts_policy_log_probs, mcts_value, mcts_stats

    @partial(jax.jit, static_argnums=(0,))
    def _act_on_batch_jit(self, observation_batch, train_state, rng):
        mcts_stats_key, action_key = jax.random.split(rng)

        # Add fake timestamp dim to make a rudimentary trajectory
        observation_trajectory_batch = pytree.expand_dims(observation_batch, axis=1)

        mcts_policy_log_probs, mcts_value, _ = self._compute_mcts_statistics(
            observation_trajectory_batch, train_state, mcts_stats_key)

        # Get rid of fake timestamp dimension
        mcts_policy_log_probs = pytree.squeeze(mcts_policy_log_probs, axis=1)
        mcts_value = pytree.squeeze(mcts_value, axis=1)

        metadata = {
            'log_mcts_action_probs': mcts_policy_log_probs,
            'mcts_state_values': mcts_value,
        }
        selected_actions = jax.random.categorical(action_key, mcts_policy_log_probs)

        return selected_actions, metadata

    def _train(self, training_batch):
        self._train_state, train_stats = self._train_jit(self._train_state, training_batch, self.next_random_key())
        train_stats = pytree.update(train_stats, self._replay_buffer.get_stats())
        return train_stats

    @partial(jax.jit, static_argnums=(0,))
    def _train_jit(self, train_state, training_batch, rng):
        def loss_function(params, rng):
            def trajectory_loss(params, trajectory, rng):
                rng, representation_key = jax.random.split(rng)

                num_unroll_steps = self._config['num_train_unroll_steps']

                trajectory_latent_states = train_state.representation_fn(
                    params, trajectory['current_state'], deterministic=False, rng=representation_key)
                # Unrolls in the end of trajectory will not have next state targets
                trajectory_latent_states_padded = jnp.concatenate(
                    [
                        trajectory_latent_states,
                        jnp.zeros(
                            shape=(num_unroll_steps + 1,) + trajectory_latent_states.shape[1:],
                            dtype=jnp.float32
                        )
                    ],
                    axis=0,
                )

                num_timestamps = trajectory_latent_states.shape[0]

                current_latent_states = trajectory_latent_states
                value_loss = 0.0
                reward_loss = 0.0
                policy_loss = 0.0
                state_similarity_loss = 0.0
                for unroll_step in range(num_unroll_steps):  # TODO: lax.fori ?
                    rng, prediction_key, dynamics_key = jax.random.split(rng, 3)

                    actions = trajectory['training_targets']['action_to_next_target'][:, unroll_step]
                    loss_mask = trajectory['training_targets']['unroll_step_mask'][:, unroll_step]
                    value_targets = trajectory['training_targets']['value'][:, unroll_step]
                    reward_targets = trajectory['training_targets']['rewards_one_hot'][:, unroll_step]
                    policy_targets = trajectory['training_targets']['policy'][:, unroll_step]
                    state_targets = jax.lax.stop_gradient(trajectory_latent_states_padded[
                        unroll_step + 1: unroll_step + 1 + num_timestamps
                    ])

                    batch_prediction_fn = jax.vmap(train_state.prediction_fn, in_axes=(None, 0, 0), out_axes=0)
                    prediction_key_batch = jax.random.split(prediction_key, num_timestamps)
                    policy_log_probs, state_values = batch_prediction_fn(
                        params, current_latent_states, prediction_key_batch)

                    batch_dynamics_fn = jax.vmap(train_state.dynamics_fn, in_axes=(None, 0, 0, 0), out_axes=0)
                    dynamics_key_batch = jax.random.split(dynamics_key, num_timestamps)
                    current_latent_states, reward_log_probs = batch_dynamics_fn(
                        params, current_latent_states, actions, dynamics_key_batch)

                    step_value_loss = rlax.l2_loss(state_values, value_targets)
                    step_reward_loss = jax.vmap(rlax.categorical_cross_entropy)(reward_targets, reward_log_probs)
                    step_policy_loss = jax.vmap(rlax.categorical_cross_entropy)(policy_targets, policy_log_probs)

                    # Mask out every timestamp for which we don't have a valid target
                    # Also apply loss scaling to make loss timestamp-independent
                    loss_scale = 1.0 / jnp.sum(loss_mask)
                    value_loss += jnp.sum(loss_mask * step_value_loss) * loss_scale
                    reward_loss += jnp.sum(loss_mask * step_reward_loss) * loss_scale
                    policy_loss += jnp.sum(loss_mask * step_policy_loss) * loss_scale

                    # See EfficientZero paper (https://arxiv.org/abs/2111.00210)
                    step_state_similarity_loss = jnp.mean(rlax.l2_loss(current_latent_states, state_targets))
                    state_similarity_loss_mask = jnp.arange(num_timestamps) < num_timestamps - unroll_step - 1
                    state_similarity_loss += (
                        jnp.sum(state_similarity_loss_mask * step_state_similarity_loss) /
                        jnp.sum(state_similarity_loss_mask)
                    )

                # Also make loss independent of num_unroll_steps
                value_loss /= num_unroll_steps
                reward_loss /= num_unroll_steps
                policy_loss /= num_unroll_steps
                state_similarity_loss /= num_unroll_steps
                loss = (
                    self._config['value_loss_weight'] * value_loss +
                    self._config['reward_loss_weight'] * reward_loss +
                    self._config['policy_loss_weight'] * policy_loss +
                    self._config['state_similarity_loss_weight'] * state_similarity_loss
                )

                return loss, {
                    'value_loss' : value_loss,
                    'reward_loss': reward_loss,
                    'policy_loss': policy_loss,
                    'state_similarity_loss': state_similarity_loss,
                    'muzero_loss': loss,
                }

            representation_key, batch_loss_key = jax.random.split(rng)

            batch_loss = jax.vmap(trajectory_loss, in_axes=(None, 0, 0), out_axes=(0, 0))
            batch_size = pytree.get_axis_dim(training_batch, axis=0)
            batch_loss_key_array = jax.random.split(batch_loss_key, batch_size)
            per_trajectory_losses, loss_details = batch_loss(params, training_batch, batch_loss_key_array)
            loss = jnp.mean(per_trajectory_losses)

            return loss, pytree.mean(loss_details)

        rng, grad_key = jax.random.split(rng)
        grad_and_stats_func = jax.grad(loss_function, argnums=0, has_aux=True)
        grads, stats = grad_and_stats_func(train_state.params, grad_key)
        train_state = train_state.apply_gradients(grads=grads)

        return train_state, stats

    @partial(jax.jit, static_argnums=(0,))
    def _reanalyze_jit(self, train_state, trajectory_batch, rng):
        # TODO: reanalyze can be done in minibatches if memory ever becomes a problem
        mcts_policy_log_probs, mcts_values, mcts_stats = self._compute_mcts_statistics(
            trajectory_batch['current_state'], train_state, rng)
        trajectory_batch = pytree.update(trajectory_batch, {
            'metadata': {
                'log_mcts_action_probs': mcts_policy_log_probs,
                'mcts_state_values': mcts_values,
            }
        })
        reanalyze_stats = {
            'reanalyze_avg_msts_policy_entropy': jnp.mean(
                jax.vmap(jax.vmap(entropy))(trajectory_batch['metadata']['log_mcts_action_probs'])
            ),
            'reanalyze_avg_mcts_state_value': jnp.mean(trajectory_batch['metadata']['mcts_state_values']),
            'reanalyze_var_mcts_state_value': jnp.var(trajectory_batch['metadata']['mcts_state_values']),
        }
        reanalyze_stats = pytree.update(reanalyze_stats, pytree.mean(mcts_stats))
        return trajectory_batch, reanalyze_stats

    def _reanalyze(self):
        trajectory_batch = self._replay_buffer.sample_trajectory_batch(self._config['reanalyze_batch_size'])
        trajectory_batch_with_mcts_stats, reanalyze_stats = self._reanalyze_jit(
            self._train_state, trajectory_batch, self.next_random_key())
        return trajectory_batch_with_mcts_stats, reanalyze_stats

    @partial(jax.jit, static_argnums=(0,))
    def _make_training_batch_jit(self, trajectory_batch_from_acting, trajectory_batch_from_reanalyze):
        def make_training_trajectory(trajectory):
            num_timestamps = pytree.get_axis_dim(trajectory, axis=0)
            num_unroll_steps = self._config['num_train_unroll_steps']
            num_actions = self.action_space.n

            # The original MuZero paper used n-step returns, but we'll go completely off-policy
            # MuZero Reanalyze paper has shown that it works (but a bit worse for some reason)
            state_values = trajectory['metadata']['mcts_state_values']

            # Allocate memory for training targets
            trajectory = pytree.update(trajectory, {
                'training_targets': {
                    'value': jnp.zeros((num_timestamps, num_unroll_steps), dtype=jnp.float32),
                    'rewards_scalar': jnp.zeros((num_timestamps, num_unroll_steps), dtype=jnp.float32),
                    'policy': jnp.zeros((num_timestamps, num_unroll_steps, num_actions), dtype=jnp.float32),
                    'action_to_next_target': jnp.zeros((num_timestamps, num_unroll_steps), dtype=jnp.int32),
                    'unroll_step_mask': jnp.zeros((num_timestamps, num_unroll_steps), dtype=jnp.bool_)
                }
            })

            # Padding will make computations at the end of trajectory more convenient
            assert num_timestamps >= num_unroll_steps, 'There will be unroll steps with no ground truth at all otherwise'
            padded_target_sources = {
                'state_values': jnp.concatenate(
                    [state_values, jnp.zeros(num_unroll_steps - 1, dtype=jnp.float32)]
                ),
                'actions': jnp.concatenate(
                    [trajectory['actions'], jnp.zeros(num_unroll_steps - 1, dtype=jnp.int32)]
                ),
                'done': jnp.concatenate(
                    [trajectory['done'], jnp.zeros(num_unroll_steps - 1, dtype=jnp.bool_)]
                ),
                'rewards_scalar': jnp.concatenate(
                    [trajectory['rewards'], jnp.zeros(num_unroll_steps - 1, dtype=jnp.float32)]
                ),
                'log_mcts_action_probs': jnp.concatenate(
                    [
                        trajectory['metadata']['log_mcts_action_probs'],
                        jnp.ones(
                            shape=(num_unroll_steps - 1, self.action_space.n), dtype=jnp.float32
                        ) * jnp.log(1.0 / self.action_space.n),
                    ],
                    axis=-2
                )
            }

            def unroll_loop_body(unroll_step, loop_state):
                trajectory, padded_target_sources, current_done = loop_state

                current_done_extra_dim = jnp.expand_dims(current_done, axis=-1)

                unroll_step_mask = jnp.arange(0, num_timestamps) < num_timestamps - unroll_step
                action_to_next_target = jax.lax.dynamic_slice_in_dim(
                    padded_target_sources['actions'], unroll_step, num_timestamps, axis=0)
                # Value and reward targets are zero after we've reached an absorbing state
                value_targets = (
                    jax.lax.dynamic_slice_in_dim(
                        padded_target_sources['state_values'], unroll_step, num_timestamps, axis=0) *
                    (1.0 - current_done)
                )
                reward_targets_scalar = (
                    jax.lax.dynamic_slice_in_dim(
                        padded_target_sources['rewards_scalar'], unroll_step, num_timestamps, axis=0) *
                    (1.0 - current_done)
                )

                # Policy target is uniform after we've reached an absorbing state
                log_mcts_action_probs_slice = jax.lax.dynamic_slice_in_dim(
                    padded_target_sources['log_mcts_action_probs'], unroll_step, num_timestamps, axis=0)
                mcts_action_probs_slice = jnp.exp(log_mcts_action_probs_slice)
                policy_target_probs = (
                    mcts_action_probs_slice * (1.0 - current_done_extra_dim) +
                    jnp.ones_like(mcts_action_probs_slice) / self.action_space.n * current_done_extra_dim
                )

                current_done = jnp.logical_or(
                    current_done,
                    jax.lax.dynamic_slice(padded_target_sources['done'], (unroll_step, ), (num_timestamps, )))

                trajectory['training_targets'] = {
                    'value': trajectory['training_targets']['value'].at[:, unroll_step].set(value_targets),
                    'rewards_scalar' : trajectory['training_targets']['rewards_scalar'].at[:, unroll_step].set(
                        reward_targets_scalar),
                    'policy': trajectory['training_targets']['policy'].at[:, unroll_step].set(policy_target_probs),
                    'action_to_next_target': trajectory['training_targets']['action_to_next_target'].at[:, unroll_step].set(
                        action_to_next_target),
                    'unroll_step_mask': trajectory['training_targets']['unroll_step_mask'].at[:, unroll_step].set(
                        unroll_step_mask),
                }

                return trajectory, padded_target_sources, current_done

            trajectory_with_targets, _, _ = jax.lax.fori_loop(
                lower=0, upper=num_unroll_steps, body_fun=unroll_loop_body,
                init_val=(trajectory, padded_target_sources, jnp.zeros_like(trajectory['done']))
            )

            # Add one-hot rewards for discrete reward prediction
            trajectory_with_targets['training_targets']['rewards_one_hot'] = discretize_onehot(
                trajectory_with_targets['training_targets']['rewards_scalar'], self._config['reward_lookup'])

            return trajectory_with_targets

        trajectory_batch = jax.tree_map(
            lambda leaf1, leaf2: jnp.concatenate([leaf1, leaf2], axis=0),
            trajectory_batch_from_acting, trajectory_batch_from_reanalyze,
        )
        return jax.vmap(make_training_trajectory)(trajectory_batch)

    def _make_next_training_batch(self, trajectory_batch_from_acting):
        trajectory_batch_from_reanalyse, reanalyze_stats = self._reanalyze()
        return (
            self._make_training_batch_jit(trajectory_batch_from_acting, trajectory_batch_from_reanalyse),
            reanalyze_stats
        )
