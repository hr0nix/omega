import functools

from functools import partial
from typing import Callable
from dataclasses import dataclass

from absl import logging

import numpy as np

import flax.struct
import flax.training.train_state
import flax.training.checkpoints
import jax.numpy as jnp
import jax.random
import jax.tree_util
import jax.experimental.host_callback
import optax
import rlax

from ..neural.optimization import clip_gradient_by_norm
from ..math import entropy, discretize_onehot, undiscretize_expected
from ..utils import pytree
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
        'num_train_steps': 1,
        'reanalyze_batch_size': 8,
        'policy_loss_weight': 1.0,
        'value_loss_weight': 1.0,
        'reward_loss_weight': 1.0,
        'state_similarity_loss_weight': 1.0,
        'max_gradient_norm': 100000.0,
        'act_deterministic': False,
        'reanalyze_deterministic': False,
        'train_deterministic': False,
        'reward_loss_priority_weight': 1.0,
        'value_loss_priority_weight': 1.0,
        'initial_priority': 1.0,
        'use_priorities': False,
        'update_next_trajectory_memory': False,
    })

    class TrainState(flax.training.train_state.TrainState):
        initial_memory_state_fn: Callable = flax.struct.field(pytree_node=False)
        representation_fn: Callable = flax.struct.field(pytree_node=False)
        dynamics_fn: Callable = flax.struct.field(pytree_node=False)
        prediction_fn: Callable = flax.struct.field(pytree_node=False)
        step_index: int = 0

    @dataclass(eq=True, frozen=True)
    class TrajectoryId(object):
        env_index: int
        step: int

    def __init__(self, *args, model_factory=NethackPerceiverMuZeroModel, replay_buffer, config=None, **kwargs):
        super(NethackMuZeroAgent, self).__init__(*args, **kwargs)

        self._config = self.CONFIG.copy(config or {})
        self._reward_lookup = {
            self._config['reward_values'][i]: i
            for i in range(len(self._config['reward_values']))
        }
        self._replay_buffer = replay_buffer
        self._current_train_step = 0

        self._build_model(model_factory)

    def _build_model(self, model_factory):
        self._model = model_factory(
            num_actions=self.action_space.n, **self._config['model_config'],
            reward_dim=len(self._reward_lookup),
            name='mu_zero_model')

        initial_memory_params = self._model.init(
            self.next_random_key(), method=self._model.initial_memory_state)
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
        model_params = merge_params(initial_memory_params, representation_params, dynamics_params, prediction_params)

        optimizer = optax.adam(learning_rate=self._config['lr'])

        self._train_state = self.TrainState.create(
            params=model_params, tx=optimizer,
            apply_fn=self._model.apply,
            initial_memory_state_fn=functools.partial(self._model.apply, method=self._model.initial_memory_state),
            representation_fn=functools.partial(self._model.apply, method=self._model.representation),
            dynamics_fn=functools.partial(self._model.apply, method=self._model.dynamics),
            prediction_fn=functools.partial(self._model.apply, method=self._model.prediction),
        )

    def _make_fake_observation(self):
        return {
            key: jnp.zeros(desc.shape, dtype=desc.dtype)
            for key, desc in self.observation_space.spaces.items()
        }

    def _make_fake_memory(self):
        memory_shape = self._model.apply({}, method=self._model.memory_shape)
        return jnp.zeros(shape=memory_shape, dtype=jnp.float32)

    def _make_fake_latent_state(self):
        latent_state_shape = self._model.apply({}, method=self._model.latent_state_shape)
        return jnp.zeros(shape=latent_state_shape, dtype=jnp.float32)

    def _make_fake_action(self):
        return jnp.zeros(shape=(), dtype=jnp.int32)

    def _make_fake_representation_inputs(self):
        return self._make_fake_memory(), self._make_fake_action(), self._make_fake_observation()

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
        # We always train on reanalysed data, fresh data is just used to fill in the replay buffer
        # Some compute might be wasted on reanalysing fresh data in the early iterations, but we don't care
        self._add_to_replay_buffer(trajectory_batch, self._current_train_step)

        stats_per_train_step = []
        for train_step in range(self._config['num_train_steps']):
            training_batch, reanalyse_stats, batch_items = self._make_next_training_batch()
            training_stats, per_trajectory_loss_details = self._train(training_batch)
            train_step_stats = pytree.update(reanalyse_stats, training_stats)

            if self._config['use_priorities']:
                self._update_replay_buffer_priorities(batch_items, per_trajectory_loss_details)

            memory_stats = self._maybe_update_next_trajectory_memory(batch_items, training_batch)
            train_step_stats = pytree.update(train_step_stats, memory_stats)

            stats_per_train_step.append(train_step_stats)

        self._current_train_step += 1

        stats = pytree.array_mean(stats_per_train_step, result_device='cpu')
        return stats

    def act_on_batch(self, observation_batch, memory_batch):
        return self._act_on_batch_jit(
            observation_batch, memory_batch, self._train_state, self.next_random_key())

    def init_memory_batch(self, batch_size):
        initial_memory_state = self._train_state.initial_memory_state_fn(self._train_state.params)
        return {
            'memory': pytree.stack([initial_memory_state] * batch_size, axis=0),
            # Assume that we've taken action #0 before the first state
            'prev_actions': jnp.zeros(batch_size, dtype=jnp.int32),
            # Consider the non-existent previous state a terminal state
            'prev_done': jnp.ones(batch_size, dtype=jnp.bool_),
        }

    def update_memory_batch(self, prev_memory, new_memory_state, actions, done):
        initial_memory_state = self._train_state.initial_memory_state_fn(self._train_state.params)
        initial_memory_state = pytree.expand_dims(initial_memory_state, axis=0)  # Add batch dim
        batch_size = new_memory_state.shape[0]
        done_memory_shaped = jnp.reshape(done, (batch_size, 1, 1))
        # Reset memory where a new episode has been started
        new_memory_state = (
            new_memory_state * (1 - done_memory_shaped) +
            initial_memory_state * done_memory_shaped
        )
        return {
            'memory': new_memory_state,
            'prev_actions': actions,
            'prev_done': done,
        }

    def _represent_trajectory(
            self, params, observation_trajectory, memory_trajectory,
            train_state, deterministic, rng):
        """
        Recurrently unrolls the representation function forwards to embed the given observation trajectory.
        """
        initial_memory_state_fn = train_state.initial_memory_state_fn
        representation_fn = functools.partial(train_state.representation_fn, deterministic=deterministic)

        def representation_loop(state, input):
            prev_memory, first_timestamp_of_the_day, rng = state
            representation_key, rng = jax.random.split(rng)

            prev_action = input['prev_action']
            prev_done = input['prev_done']
            cur_observation = input['observation']

            # Reset the memory if this the current state is the first state of an episode
            initial_memory_state = initial_memory_state_fn(params)
            prev_memory = prev_memory * (1 - prev_done) + initial_memory_state * prev_done

            # Recurrently embed the observation and compute the updated memory
            latent_observation, updated_memory = representation_fn(
                params, prev_memory, prev_action, cur_observation, representation_key)

            return (updated_memory, False, rng), (latent_observation, updated_memory)

        num_timestamps = pytree.get_axis_dim(observation_trajectory, axis=0)
        # Recompute memory within the trajectory but use a fixed initial state
        initial_memory = memory_trajectory['memory'][0]
        _, (latent_state_trajectory, updated_memory_trajectory) = jax.lax.scan(
            f=representation_loop,
            init=(initial_memory, True, rng),
            xs={
                'prev_action': memory_trajectory['prev_actions'],
                'prev_done': memory_trajectory['prev_done'],
                'observation': observation_trajectory,
            },
            length=num_timestamps,
        )

        return latent_state_trajectory, updated_memory_trajectory

    def _add_to_replay_buffer(self, trajectory_batch, current_train_step):
        # Don't want multiple reads from GPU memory and replay buffer stores everything in RAM anyway
        trajectory_batch = pytree.to_cpu(trajectory_batch)
        batch_size = pytree.get_axis_dim(trajectory_batch, 0)
        for env_idx in range(batch_size):
            trajectory = pytree.batch_dim_slice(trajectory_batch, env_idx)
            priority = self._config['initial_priority'] if self._config['use_priorities'] else None
            self._replay_buffer.add_trajectory(
                trajectory_id=self.TrajectoryId(env_index=env_idx, step=current_train_step),
                trajectory=trajectory,
                priority=priority,
                current_step=current_train_step
            )

    def _update_replay_buffer_priorities(self, replayed_items, trajectory_loss_details):
        reward_loss = pytree.to_cpu(trajectory_loss_details['reward_loss'])
        value_loss = pytree.to_cpu(trajectory_loss_details['value_loss'])
        priorities = (
            self._config['reward_loss_priority_weight'] * reward_loss +
            self._config['value_loss_priority_weight'] * value_loss
        )
        for index, item in enumerate(replayed_items):
            self._replay_buffer.update_priority(item.id, priorities[index])

    def _maybe_update_next_trajectory_memory(self, replayed_items, training_batch):
        # Make sure terminal states are taken into account when updating memory
        updated_memory = self.update_memory_batch(
            pytree.timestamp_dim_slice(training_batch['memory_before'], slice_idx=-1),
            pytree.timestamp_dim_slice(training_batch['mcts_reanalyze']['memory_state_after'], slice_idx=-1),
            pytree.timestamp_dim_slice(training_batch['actions'], slice_idx=-1),
            pytree.timestamp_dim_slice(training_batch['done'], slice_idx=-1),
        )
        updated_memory_state_cpu = pytree.to_cpu(updated_memory['memory'])

        memory_diffs = []
        for index, item in enumerate(replayed_items):
            next_trajectory_id = self.TrajectoryId(env_index=item.id.env_index, step=item.id.step + 1)
            next_trajectory_item = self._replay_buffer.find_trajectory(next_trajectory_id)
            if next_trajectory_item is None:
                # This is the most fresh trajectory
                assert item.id.step == self._current_train_step
                continue
            # TODO: it's not nice to mix numpy and jax arrays, switch to using jax on CPU as well
            memory_state_copy = next_trajectory_item.trajectory['memory_before']['memory'].copy()
            diff_sqr = np.mean((memory_state_copy - updated_memory_state_cpu[index, -1]) ** 2)
            memory_diffs.append(diff_sqr)
            memory_state_copy[0] = updated_memory_state_cpu[index, -1]
            if self._config['update_next_trajectory_memory']:
                next_trajectory_item.trajectory['memory_before']['memory'] = memory_state_copy

        stats = {}
        if len(memory_diffs) > 0:
            stats = pytree.update(stats, {
                'avg_memory_update_diff_sqr': np.mean(memory_diffs)
            })
        return stats

    def _compute_mcts_statistics(
            self, observation_trajectory_batch, memory_trajectory_batch, train_state, deterministic, rng):
        representation_key, mcts_key = jax.random.split(rng)

        batch_size, num_timestamps = jax.tree_leaves(observation_trajectory_batch)[0].shape[:2]

        represent_trajectory_batch_fn = jax.vmap(self._represent_trajectory, in_axes=(None, 0, 0, None, None, 0))
        representation_key_batch = jax.random.split(representation_key, batch_size)
        latent_state_trajectory_batch, updated_memory_trajectory_batch = represent_trajectory_batch_fn(
            train_state.params,
            observation_trajectory_batch, memory_trajectory_batch,
            train_state, deterministic, representation_key_batch)

        def dynamics_fn(params, state, action, rng, deterministic):
            next_state, reward_logits = train_state.dynamics_fn(
                params, state, action, rng, deterministic=deterministic)
            # Convert reward back to continuous form
            return next_state, undiscretize_expected(reward_logits, self._reward_lookup)

        mcts_func = functools.partial(
            mcts,
            num_actions=self.action_space.n,
            prediction_fn=jax.tree_util.Partial(
                train_state.prediction_fn, train_state.params, deterministic=deterministic),
            dynamics_fn=jax.tree_util.Partial(
                dynamics_fn, train_state.params, deterministic=deterministic),
            discount_factor=self._config['discount_factor'],
            num_simulations=self._config['num_mcts_simulations'],
            puct_c1=self._config['mcts_puct_c1'],
            dirichlet_noise_alpha=self._config['mcts_dirichlet_noise_alpha'],
            root_exploration_fraction=self._config['mcts_root_exploration_fraction'],
        )
        trajectory_mcts = jax.vmap(mcts_func)
        trajectory_batch_mcts = jax.vmap(trajectory_mcts)
        mcts_key_batch = jax.random.split(mcts_key, batch_size * num_timestamps).reshape(batch_size, num_timestamps, 2)
        mcts_policy_log_probs, mcts_value, mcts_stats = trajectory_batch_mcts(
            latent_state_trajectory_batch, mcts_key_batch)

        return updated_memory_trajectory_batch, mcts_policy_log_probs, mcts_value, mcts_stats

    @partial(jax.jit, static_argnums=(0,))
    def _act_on_batch_jit(self, observation_batch, memory_batch, train_state, rng):
        mcts_stats_key, action_key = jax.random.split(rng)

        # Add fake timestamp dim to make a rudimentary trajectory
        observation_trajectory_batch = pytree.expand_dims(observation_batch, axis=1, result_device='gpu')
        memory_trajectory_batch = pytree.expand_dims(memory_batch, axis=1, result_device='gpu')

        updated_memory, mcts_policy_log_probs, mcts_value, _ = self._compute_mcts_statistics(
            observation_trajectory_batch, memory_trajectory_batch,
            train_state, self._config['act_deterministic'], mcts_stats_key)

        # Get rid of the fake timestamp dimension that we've added before
        mcts_policy_log_probs = pytree.squeeze(mcts_policy_log_probs, axis=1)
        mcts_value = pytree.squeeze(mcts_value, axis=1)
        updated_memory = pytree.squeeze(updated_memory, axis=1)

        # Choose actions to execute in the environment
        selected_actions = jax.random.categorical(action_key, mcts_policy_log_probs)

        metadata = {
            'memory_state_after': updated_memory,
            'log_mcts_action_probs': mcts_policy_log_probs,
            'mcts_state_values': mcts_value,
        }

        return selected_actions, metadata

    def _train(self, training_batch):
        self._train_state, train_stats, per_trajectory_loss_details = self._train_jit(
            self._train_state, training_batch, self.next_random_key())
        train_stats = pytree.update(train_stats, self._replay_buffer.get_stats())
        return train_stats, per_trajectory_loss_details

    @partial(jax.jit, static_argnums=(0,))
    def _train_jit(self, train_state, training_batch, rng):
        def loss_function(params, rng):
            def trajectory_loss(params, trajectory, rng):
                rng, representation_key = jax.random.split(rng)
                deterministic = self._config['train_deterministic']

                num_unroll_steps = self._config['num_train_unroll_steps']

                trajectory_latent_states, _ = self._represent_trajectory(
                    params, trajectory['current_state'], trajectory['memory_before'],
                    train_state, deterministic, representation_key)
                trajectory_latent_states_padded = jnp.concatenate(
                    [
                        trajectory_latent_states,
                        jnp.zeros(
                            shape=(num_unroll_steps,) + trajectory_latent_states.shape[1:], dtype=jnp.float32
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

                    actions = trajectory['training_targets']['actions'][:, unroll_step]
                    loss_mask = trajectory['training_targets']['unroll_step_mask'][:, unroll_step]
                    next_state_is_terminal_or_after = trajectory['training_targets']['next_state_is_terminal_or_after'][:, unroll_step]
                    value_targets = trajectory['training_targets']['value'][:, unroll_step]
                    reward_targets = trajectory['training_targets']['rewards_one_hot'][:, unroll_step]
                    policy_targets = trajectory['training_targets']['policy'][:, unroll_step]
                    state_targets = trajectory_latent_states_padded[unroll_step + 1: unroll_step + 1 + num_timestamps]

                    prediction_fn = functools.partial(train_state.prediction_fn, deterministic=deterministic)
                    batch_prediction_fn = jax.vmap(prediction_fn, in_axes=(None, 0, 0), out_axes=0)
                    prediction_key_batch = jax.random.split(prediction_key, num_timestamps)
                    policy_log_probs, state_values = batch_prediction_fn(
                        params, current_latent_states, prediction_key_batch)

                    dynamics_fn = functools.partial(train_state.dynamics_fn, deterministic=deterministic)
                    batch_dynamics_fn = jax.vmap(dynamics_fn, in_axes=(None, 0, 0, 0), out_axes=0)
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
                    step_state_similarity_loss = jnp.mean(
                        rlax.l2_loss(current_latent_states, state_targets),
                        axis=(-2, -1))
                    # We don't have targets at the end of a trajectory
                    state_similarity_loss_mask = jnp.arange(num_timestamps) < num_timestamps - unroll_step - 1
                    # We also don't have next state targets for terminal states or anything after
                    state_similarity_loss_mask *= 1.0 - next_state_is_terminal_or_after
                    state_similarity_loss += (
                        jnp.sum(state_similarity_loss_mask * step_state_similarity_loss) /
                        (jnp.sum(state_similarity_loss_mask) + 1e-10)  # Just in case
                    )

                # Make loss independent of num_unroll_steps
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
            per_trajectory_losses, per_trajectory_loss_details = batch_loss(
                params, training_batch, batch_loss_key_array)
            loss = jnp.mean(per_trajectory_losses)

            return loss, per_trajectory_loss_details

        rng, grad_key = jax.random.split(rng)
        grad_and_loss_details_func = jax.grad(loss_function, argnums=0, has_aux=True)
        grads, per_trajectory_loss_details = grad_and_loss_details_func(train_state.params, grad_key)
        grads, grad_norm = clip_gradient_by_norm(grads, self._config['max_gradient_norm'], return_norm=True)
        train_state = train_state.apply_gradients(grads=grads)

        stats = pytree.mean(per_trajectory_loss_details)
        stats = pytree.update(stats, {
            'gradient_norm': grad_norm,
        })

        return train_state, stats, per_trajectory_loss_details

    # TODO: for some reason total execution time of this function
    # TODO: is 200 ms more than _make_next_training_batch_jit + sample_trajectory_batch
    def _make_next_training_batch(self):
        trajectory_batch_items = self._replay_buffer.sample_trajectory_batch(self._config['reanalyze_batch_size'])
        trajectory_batch = pytree.stack([item.trajectory for item in trajectory_batch_items], axis=0)

        training_batch, reanalyse_stats = self._make_next_training_batch_jit(
            self._train_state, trajectory_batch, self.next_random_key())

        return training_batch, reanalyse_stats, trajectory_batch_items

    @partial(jax.jit, static_argnums=(0,))
    def _make_next_training_batch_jit(self, train_state, trajectory_batch, rng):
        trajectory_batch_with_mcts_stats, reanalyze_stats = self._reanalyze(train_state, trajectory_batch, rng)
        trajectory_batch_with_training_targets = self._prepare_training_targets(trajectory_batch_with_mcts_stats)
        return trajectory_batch_with_training_targets, reanalyze_stats

    def _reanalyze(self, train_state, trajectory_batch, rng):
        # TODO: reanalyze can be done in minibatches if memory ever becomes a problem
        updated_memory, mcts_policy_log_probs, mcts_values, mcts_stats = self._compute_mcts_statistics(
            trajectory_batch['current_state'], trajectory_batch['memory_before'],
            train_state, self._config['reanalyze_deterministic'], rng)
        trajectory_batch = pytree.update(trajectory_batch, {
            'mcts_reanalyze': {
                'log_action_probs': mcts_policy_log_probs,
                'state_values': mcts_values,
                'memory_state_after': updated_memory,
            },
        })
        reanalyze_stats = {
            'reanalyze_avg_msts_policy_entropy': jnp.mean(
                jax.vmap(jax.vmap(entropy))(trajectory_batch['mcts_reanalyze']['log_action_probs'])
            ),
            'reanalyze_avg_mcts_state_value': jnp.mean(trajectory_batch['mcts_reanalyze']['state_values']),
            'reanalyze_var_mcts_state_value': jnp.var(trajectory_batch['mcts_reanalyze']['state_values']),
        }
        reanalyze_stats = pytree.update(reanalyze_stats, pytree.mean(mcts_stats))
        return trajectory_batch, reanalyze_stats

    def _prepare_training_targets(self, trajectory_batch):
        def make_training_trajectory(trajectory):
            num_timestamps = pytree.get_axis_dim(trajectory, axis=0)
            num_unroll_steps = self._config['num_train_unroll_steps']
            num_actions = self.action_space.n

            # The original MuZero paper used n-step returns, but we'll go completely off-policy.
            # MuZero Reanalyze paper has shown that it works, but a bit worse for some reason. Works fine for us.
            state_values = trajectory['mcts_reanalyze']['state_values']

            # Allocate memory for training targets
            trajectory = pytree.update(trajectory, {
                'training_targets': {
                    'value': jnp.zeros((num_timestamps, num_unroll_steps), dtype=jnp.float32),
                    'rewards_scalar': jnp.zeros((num_timestamps, num_unroll_steps), dtype=jnp.float32),
                    'policy': jnp.zeros((num_timestamps, num_unroll_steps, num_actions), dtype=jnp.float32),
                    'actions': jnp.zeros((num_timestamps, num_unroll_steps), dtype=jnp.int32),
                    'unroll_step_mask': jnp.zeros((num_timestamps, num_unroll_steps), dtype=jnp.bool_),
                    'next_state_is_terminal_or_after': jnp.zeros((num_timestamps, num_unroll_steps), dtype=jnp.bool_),
                }
            })

            # Padding will make computations at the end of a trajectory more convenient
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
                        trajectory['mcts_reanalyze']['log_action_probs'],
                        jnp.full(
                            shape=(num_unroll_steps - 1, self.action_space.n),
                            fill_value=jnp.log(1.0 / self.action_space.n)
                        ),
                    ],
                    axis=-2
                )
            }

            def unroll_loop_body(unroll_step, loop_state):
                trajectory, padded_target_sources, state_is_terminal_or_after = loop_state

                state_is_terminal_or_after_extra_dim = jnp.expand_dims(state_is_terminal_or_after, axis=-1)

                unroll_step_mask = jnp.arange(0, num_timestamps) < num_timestamps - unroll_step
                actions = jax.lax.dynamic_slice_in_dim(
                    padded_target_sources['actions'], unroll_step, num_timestamps, axis=0)
                # Value and reward targets are zero after we've reached an absorbing state
                value_targets = (
                    jax.lax.dynamic_slice_in_dim(
                        padded_target_sources['state_values'], unroll_step, num_timestamps, axis=0) *
                    (1.0 - state_is_terminal_or_after)
                )
                reward_targets_scalar = (
                    jax.lax.dynamic_slice_in_dim(
                        padded_target_sources['rewards_scalar'], unroll_step, num_timestamps, axis=0) *
                    (1.0 - state_is_terminal_or_after)
                )

                # Policy target is uniform after we've reached an absorbing state
                log_mcts_action_probs_slice = jax.lax.dynamic_slice_in_dim(
                    padded_target_sources['log_mcts_action_probs'], unroll_step, num_timestamps, axis=0)
                mcts_action_probs_slice = jnp.exp(log_mcts_action_probs_slice)
                policy_target_probs = (
                    mcts_action_probs_slice * (1.0 - state_is_terminal_or_after_extra_dim) +
                    jnp.full_like(mcts_action_probs_slice, 1.0 / self.action_space.n) * state_is_terminal_or_after_extra_dim
                )

                next_state_is_terminal_or_after = jnp.logical_or(
                    state_is_terminal_or_after,
                    jax.lax.dynamic_slice_in_dim(padded_target_sources['done'], unroll_step, num_timestamps, axis=0)
                )

                trajectory['training_targets'] = {
                    'value': trajectory['training_targets']['value'].at[:, unroll_step].set(value_targets),
                    'rewards_scalar' : trajectory['training_targets']['rewards_scalar'].at[:, unroll_step].set(
                        reward_targets_scalar),
                    'policy': trajectory['training_targets']['policy'].at[:, unroll_step].set(policy_target_probs),
                    'actions': trajectory['training_targets']['actions'].at[:, unroll_step].set(
                        actions),
                    'unroll_step_mask': trajectory['training_targets']['unroll_step_mask'].at[:, unroll_step].set(
                        unroll_step_mask),
                    'next_state_is_terminal_or_after':
                        trajectory['training_targets']['next_state_is_terminal_or_after'].at[:, unroll_step].set(
                            next_state_is_terminal_or_after),
                }

                return trajectory, padded_target_sources, next_state_is_terminal_or_after

            trajectory_with_targets, _, _ = jax.lax.fori_loop(
                lower=0, upper=num_unroll_steps, body_fun=unroll_loop_body,
                init_val=(trajectory, padded_target_sources, jnp.zeros_like(trajectory['done']))
            )

            # Add one-hot rewards for discrete reward prediction
            trajectory_with_targets['training_targets']['rewards_one_hot'] = discretize_onehot(
                trajectory_with_targets['training_targets']['rewards_scalar'], self._reward_lookup)

            return trajectory_with_targets

        return jax.vmap(make_training_trajectory)(trajectory_batch)
