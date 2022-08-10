import functools

import chex
import math
import os

from functools import partial
from typing import Callable
from dataclasses import dataclass

from absl import logging

import flax.struct
import flax.training.train_state
import flax.training.checkpoints
import jax.numpy as jnp
import jax.random
import jax.tree_util
import jax.experimental.host_callback
import optax
import rlax

from ..math import entropy
from ..utils import pytree
from ..utils.flax import merge_params
from ..utils.profiling import timeit
from ..models.nethack_muzero import NethackPerceiverMuZeroModel
from ..mcts.muzero import mcts
from .trainable_agent import JaxTrainableAgentBase


class NethackMuZeroAgent(JaxTrainableAgentBase):
    CONFIG = flax.core.frozen_dict.FrozenDict({
        'lr': 1e-3,
        'use_adaptive_lr': False,
        'num_lr_warmup_steps': 0,
        'weight_decay': 0.0,
        'discount_factor': 0.99,
        'model_config': {},
        'num_mcts_simulations': 30,
        'mcts_puct_c1': 1.25,
        'mcts_dirichlet_noise_alpha': 0.2,
        'mcts_root_exploration_fraction': 0.2,
        'mcts_search_policy': 'puct',
        'mcts_result_policy': 'visit_count',
        'num_train_unroll_steps': 5,
        'num_train_steps': 1,
        'reanalyze_batch_size': 8,
        'warmup_days': 0,
        'value_reward_bins': 64,
        'value_reward_min_max': (-1.0, 1.0),
        'chance_outcome_commitment_loss_weight': 50.0,
        'chance_outcome_prediction_loss_weight': 1.0,
        'policy_loss_weight': 1.0,
        'value_loss_weight': 1.0,
        'afterstate_value_loss_weight': 1.0,
        'reward_loss_weight': 10.0,
        'state_similarity_loss_weight': 0.001,
        'state_similarity_loss_stop_gradient': False,
        'max_gradient_norm': 1000.0,
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
        chance_outcome_encoder_fn: Callable = flax.struct.field(pytree_node=False)
        representation_fn: Callable = flax.struct.field(pytree_node=False)
        dynamics_fn: Callable = flax.struct.field(pytree_node=False)
        afterstate_dynamics_fn: Callable = flax.struct.field(pytree_node=False)
        prediction_fn: Callable = flax.struct.field(pytree_node=False)
        afterstate_prediction_fn: Callable = flax.struct.field(pytree_node=False)
        step_index: int = 0

    @dataclass(eq=True, frozen=True)
    class TrajectoryId(object):
        env_index: int
        step: int

    def __init__(self, *args, model_factory=NethackPerceiverMuZeroModel, replay_buffer, config=None, **kwargs):
        super(NethackMuZeroAgent, self).__init__(*args, **kwargs)

        self._config = self.CONFIG.copy(config or {})
        self._replay_buffer = replay_buffer
        self._current_train_step = 0

        self._build_model(model_factory)

    def _init_model_params(self):
        initial_memory_params = self._model.init(
            {'params': self.next_random_key()},
            method=self._model.initial_memory_state)
        chance_outcome_encoder_params = self._model.init(
            {'dropout': self.next_random_key(), 'params': self.next_random_key()},
            *self._make_fake_chance_outcome_encoder_inputs(),
            method=self._model.chance_outcome_encoder, deterministic=False)
        representation_params = self._model.init(
            {'dropout': self.next_random_key(), 'params': self.next_random_key()},
            *self._make_fake_representation_inputs(),
            method=self._model.representation, deterministic=False)
        dynamics_params = self._model.init(
            {'dropout': self.next_random_key(), 'params': self.next_random_key()},
            *self._make_fake_dynamics_inputs(),
            method=self._model.dynamics, deterministic=False)
        afterstate_dynamics_params = self._model.init(
            {'dropout': self.next_random_key(), 'params': self.next_random_key()},
            *self._make_fake_afterstate_dynamics_inputs(),
            method=self._model.afterstate_dynamics, deterministic=False)
        prediction_params = self._model.init(
            {'dropout': self.next_random_key(), 'params': self.next_random_key()},
            *self._make_fake_prediction_inputs(),
            method=self._model.prediction, deterministic=False)
        afterstate_prediction_params = self._model.init(
            {'dropout': self.next_random_key(), 'params': self.next_random_key()},
            *self._make_fake_afterstate_prediction_inputs(),
            method=self._model.afterstate_prediction, deterministic=False)

        # Merge params from different initializations, some values will be overridden
        return merge_params(
            initial_memory_params, chance_outcome_encoder_params,
            representation_params, dynamics_params, afterstate_dynamics_params,
            prediction_params, afterstate_prediction_params,
        )

    def _build_model(self, model_factory):
        self._model = model_factory(
            num_actions=self.action_space.n, value_reward_dim=self._config['value_reward_bins'], name='muzero_model',
            **self._config['model_config'])
        model_params = self._init_model_params()
        optimizer = self._make_optimizer()
        self._train_state = self.TrainState.create(
            params=model_params,
            tx=optimizer,
            apply_fn=self._model.apply,
            initial_memory_state_fn=functools.partial(
                self._model.apply, method=self._model.initial_memory_state),
            chance_outcome_encoder_fn=functools.partial(
                self._model.apply, method=self._model.chance_outcome_encoder),
            representation_fn=functools.partial(
                self._model.apply, method=self._model.representation),
            dynamics_fn=functools.partial(
                self._model.apply, method=self._model.dynamics),
            afterstate_dynamics_fn=functools.partial(
                self._model.apply, method=self._model.afterstate_dynamics),
            prediction_fn=functools.partial(
                self._model.apply, method=self._model.prediction),
            afterstate_prediction_fn=functools.partial(
                self._model.apply, method=self._model.afterstate_prediction),
        )

    def _make_optimizer(self):
        lr = self._config['lr']
        if self._config['use_adaptive_lr']:
            lr *= math.sqrt(self._config['reanalyze_batch_size'])
            logging.info(f'Using adaptive learning rate: {lr}')

        lr_schedules = [
            optax.linear_schedule(init_value=0.0, end_value=lr, transition_steps=self._config['num_lr_warmup_steps']),
            optax.constant_schedule(lr),
        ]
        lr_schedule = optax.join_schedules(
            schedules=lr_schedules, boundaries=[self._config['num_lr_warmup_steps']]
        )

        return optax.chain(
            optax.clip_by_global_norm(self._config['max_gradient_norm']),
            optax.adamw(learning_rate=lr_schedule, weight_decay=self._config['weight_decay']),
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

    def _make_fake_chance_outcome_one_hot(self):
        return jax.nn.one_hot(0, num_classes=self._model.num_chance_outcomes, dtype=jnp.float32)

    def _make_fake_chance_outcome_encoder_inputs(self):
        return self._make_fake_latent_state(),

    def _make_fake_representation_inputs(self):
        return self._make_fake_memory(), self._make_fake_action(), self._make_fake_observation()

    def _make_fake_dynamics_inputs(self):
        return self._make_fake_latent_state(), self._make_fake_chance_outcome_one_hot()

    def _make_fake_afterstate_dynamics_inputs(self):
        return self._make_fake_latent_state(), self._make_fake_action()

    def _make_fake_prediction_inputs(self):
        return self._make_fake_latent_state(),

    def _make_fake_afterstate_prediction_inputs(self):
        return self._make_fake_latent_state(),

    def try_load_from_checkpoint(self, path):
        checkpoint_path = flax.training.checkpoints.latest_checkpoint(path)
        if checkpoint_path is None:
            logging.warning(f'No checkpoints available at {path}')
            return 0
        else:
            logging.info(f'State will be loaded from checkpoint {checkpoint_path}')
            self._train_state = flax.training.checkpoints.restore_checkpoint(checkpoint_path, self._train_state)
            self._current_train_step = self._train_state.step_index
            self._replay_buffer.load(
                os.path.join(path, self._get_replay_buffer_checkpoint_filename_prefix()))
            return self._current_train_step

    def save_to_checkpoint(self, checkpoint_path):
        self._train_state = self._train_state.replace(step_index=self._current_train_step)
        flax.training.checkpoints.save_checkpoint(
            checkpoint_path, self._train_state, step=self._current_train_step, keep=1, overwrite=True)
        self._replay_buffer.save(
            os.path.join(checkpoint_path, self._get_replay_buffer_checkpoint_filename_prefix()))
        self._remove_old_replay_buffer_checkpoints(checkpoint_path)

    def _remove_old_replay_buffer_checkpoints(self, checkpoint_path):
        current_checkpoint_filename_prefix = self._get_replay_buffer_checkpoint_filename_prefix()
        for checkpoint_filename in os.listdir(checkpoint_path):
            if checkpoint_filename.startswith('replay_buffer.') and not checkpoint_filename.startswith(
                    current_checkpoint_filename_prefix):
                full_checkpoint_filename = os.path.join(checkpoint_path, checkpoint_filename)
                logging.info(f'Removing old replay buffer checkpoint file {full_checkpoint_filename}')
                os.remove(full_checkpoint_filename)

    def _get_replay_buffer_checkpoint_filename_prefix(self):
        return f'replay_buffer.step_{self._current_train_step}'

    @partial(jax.jit, static_argnums=0)
    def _get_current_batch_stats_jit(self, trajectory_batch):
        return {
            'act_avg_mcts_policy_entropy': jnp.mean(
                jax.vmap(jax.vmap(entropy))(trajectory_batch['act_metadata']['log_mcts_action_probs'])
            ),
            'act_avg_mcts_state_value': jnp.mean(trajectory_batch['act_metadata']['mcts_state_values']),
            'act_var_mcts_state_value': jnp.var(trajectory_batch['act_metadata']['mcts_state_values']),
        }

    @timeit
    def train_on_batch(self, trajectory_batch):
        # We always train on reanalysed data, fresh data is just used to fill in the replay buffer
        # Some compute might be wasted on reanalysing fresh data in the early iterations, but we don't care
        current_batch_stats = self._get_current_batch_stats_jit(trajectory_batch)
        self._add_to_replay_buffer(trajectory_batch)

        stats = pytree.to_numpy(current_batch_stats)

        if self._current_train_step >= self._config['warmup_days']:
            stats_per_train_step = []
            for train_step in range(self._config['num_train_steps']):
                training_batch, reanalyse_stats, batch_items = self._make_next_training_batch()
                training_stats, per_trajectory_loss_details = self._train(training_batch)
                train_step_stats = pytree.update(reanalyse_stats, training_stats)

                if self._config['use_priorities']:
                    self._update_replay_buffer_priorities(batch_items, per_trajectory_loss_details)

                if self._config['update_next_trajectory_memory']:
                    memory_stats = self._update_next_trajectory_memory(batch_items, training_batch)
                    train_step_stats = pytree.update(train_step_stats, memory_stats)

                stats_per_train_step.append(train_step_stats)

            stats = pytree.update(stats, pytree.array_mean(stats_per_train_step, result_backend='numpy'))

        self._current_train_step += 1
        return stats

    @timeit
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

    @timeit
    def update_memory_batch(self, prev_memory, new_memory_state, actions, done):
        return self._update_memory_batch_jit(self._train_state, new_memory_state, actions, done)

    @timeit
    @partial(jax.jit, static_argnums=0)
    def _update_memory_batch_jit(self, train_state, new_memory_state, actions, done):
        initial_memory_state = train_state.initial_memory_state_fn(train_state.params)
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

    def _encode_value_or_reward(self, value_or_reward):
        return rlax.transform_to_2hot(
            value_or_reward,
            min_value=self._config['value_reward_min_max'][0],
            max_value=self._config['value_reward_min_max'][1],
            num_bins=self._config['value_reward_bins'],
        )

    def _decode_value_or_reward(self, value_or_reward_log_probs):
        chex.assert_axis_dimension(value_or_reward_log_probs, -1, self._config['value_reward_bins'])

        encoded_value_or_reward_probs = jax.nn.softmax(value_or_reward_log_probs)
        result = rlax.transform_from_2hot(
            encoded_value_or_reward_probs,
            min_value=self._config['value_reward_min_max'][0],
            max_value=self._config['value_reward_min_max'][1],
            num_bins=self._config['value_reward_bins'],
        )
        return jnp.squeeze(result, axis=-1)

    @staticmethod
    def _represent_trajectory(params, observation_trajectory, memory_trajectory, train_state, deterministic, rng):
        """
        Recurrently unrolls the representation function forwards starting from the initial memory state
        to embed the given observation trajectory.
        """
        initial_memory_state_fn = train_state.initial_memory_state_fn
        representation_fn = functools.partial(train_state.representation_fn, deterministic=deterministic)

        def representation_loop(state, input):
            rng, prev_memory, first_timestamp_of_the_day = state

            prev_action = input['prev_action']
            prev_done = input['prev_done']
            cur_observation = input['observation']

            # Reset the memory if this the current state is the first state of an episode
            initial_memory_state = initial_memory_state_fn(params)
            prev_memory = prev_memory * (1 - prev_done) + initial_memory_state * prev_done

            # Recurrently embed the observation and compute the updated memory
            rng, representation_key = jax.random.split(rng)
            latent_observation, updated_memory = representation_fn(
                params, prev_memory, prev_action, cur_observation, rngs={'dropout': representation_key})

            return (rng, updated_memory, False), (latent_observation, updated_memory)

        num_timestamps = pytree.get_axis_dim(observation_trajectory, axis=0)
        # Recompute memory within the trajectory but use a fixed initial state
        initial_memory = memory_trajectory['memory'][0]
        _, (latent_state_trajectory, updated_memory_trajectory) = jax.lax.scan(
            f=representation_loop,
            init=(rng, initial_memory, True),
            xs={
                'prev_action': memory_trajectory['prev_actions'],
                'prev_done': memory_trajectory['prev_done'],
                'observation': observation_trajectory,
            },
            length=num_timestamps,
        )

        return latent_state_trajectory, updated_memory_trajectory

    @timeit
    def _add_to_replay_buffer(self, trajectory_batch):
        # Make CPU versions of tensors that might be accessed by the replay buffer
        rewards_cpu = pytree.to_numpy(trajectory_batch['rewards'])

        batch_size = pytree.get_axis_dim(trajectory_batch, 0)
        trajectories = jax.jit(pytree.split, static_argnames=['size', 'axis'])(
            trajectory_batch, batch_size, axis=0)  # Jit makes splitting faster

        for env_idx in range(batch_size):
            # Add CPU tensors to each trajectory
            trajectory = pytree.update(trajectories[env_idx], {
                'rewards_cpu': rewards_cpu[env_idx],
            })

            priority = self._config['initial_priority'] if self._config['use_priorities'] else None
            self._replay_buffer.add_trajectory(
                trajectory_id=self.TrajectoryId(env_index=env_idx, step=self._current_train_step),
                trajectory=trajectory,
                priority=priority,
                current_step=self._current_train_step
            )

    @timeit
    def _update_replay_buffer_priorities(self, replayed_items, trajectory_loss_details):
        # TODO: If switching back to using priorities, think about avoiding device to host copying here
        reward_loss = pytree.to_numpy(trajectory_loss_details['reward_loss'])
        value_loss = pytree.to_numpy(trajectory_loss_details['value_loss'])
        priorities = (
            self._config['reward_loss_priority_weight'] * reward_loss +
            self._config['value_loss_priority_weight'] * value_loss
        )
        for index, item in enumerate(replayed_items):
            self._replay_buffer.update_priority(item.id, priorities[index])

    @timeit
    @partial(jax.jit, static_argnums=0)
    def _get_updated_memory_state_after_last_ts_batch_jit(self, trajectory_batch):
        # Make sure terminal states are taken into account when updating memory
        updated_memory_after_last_ts_batch = self.update_memory_batch(
            pytree.timestamp_dim_slice(trajectory_batch['memory_before'], slice_idx=-1),
            pytree.timestamp_dim_slice(trajectory_batch['mcts_reanalyze']['memory_state_after'], slice_idx=-1),
            pytree.timestamp_dim_slice(trajectory_batch['actions'], slice_idx=-1),
            pytree.timestamp_dim_slice(trajectory_batch['done'], slice_idx=-1),
        )
        return updated_memory_after_last_ts_batch['memory']

    @timeit
    @partial(jax.jit, static_argnums=0)
    def _update_initial_trajectory_memory_jit(self, memories_to_update, update_source_ids, update_source):
        def updater(memory_to_update, update_source_id):
            update = update_source[update_source_id]
            memory_abs_diff = jnp.mean(jnp.abs(memory_to_update[0] - update))
            updated_memory = memory_to_update.at[0].set(update)
            return updated_memory, memory_abs_diff
        updated = jax.tree_map(updater, memories_to_update, update_source_ids)
        updated_memories, memory_abs_diffs = jax.tree_transpose(
            inner_treedef=jax.tree_structure([_ for _ in range(2)]),
            outer_treedef=jax.tree_structure(memories_to_update),
            pytree_to_transpose=updated
        )
        return updated_memories, pytree.array_mean(memory_abs_diffs)

    @timeit
    def _update_next_trajectory_memory(self, replayed_items, training_batch):
        """
        Given that we have an updated memory for each trajectory in the batch,
        we can update initial memories of the trajectories that temporally follow the batch
        (provided that they still are in the replay buffer).
        """
        updated_memory_state_after_last_ts_batch = self._get_updated_memory_state_after_last_ts_batch_jit(
            training_batch)

        items_to_update = []
        memories_to_update = []
        update_source_ids = []
        for batch_index, trajectory_item in enumerate(replayed_items):
            next_trajectory_id = self.TrajectoryId(
                env_index=trajectory_item.id.env_index, step=trajectory_item.id.step + 1)
            next_trajectory_item = self._replay_buffer.find_trajectory(next_trajectory_id)
            if next_trajectory_item is None:
                # Either this is the freshest trajectory, or the next trajectory
                # has been evicted (this can happen with some replay buffer types, i.e. clustering replay).
                continue

            items_to_update.append(next_trajectory_item)
            memories_to_update.append(next_trajectory_item.trajectory['memory_before']['memory'])
            update_source_ids.append(batch_index)

        stats = {}
        if len(memories_to_update) > 0:
            updated_memories, avg_memory_update_abs_diff = self._update_initial_trajectory_memory_jit(
                memories_to_update, update_source_ids, updated_memory_state_after_last_ts_batch)

            for item, updated_memory in zip(items_to_update, updated_memories):
                item.trajectory['memory_before']['memory'] = updated_memory

            stats = pytree.update(stats, {
                'avg_memory_update_abs_diff': avg_memory_update_abs_diff
            })

        return stats

    def _compute_mcts_statistics(
            self, observation_trajectory_batch, memory_trajectory_batch, train_state, deterministic, rng):
        batch_size, num_timestamps = jax.tree_leaves(observation_trajectory_batch)[0].shape[:2]

        rng, represent_trajectory_key = jax.random.split(rng)
        represent_trajectory_key_batch = jax.random.split(represent_trajectory_key, batch_size)
        represent_trajectory_batch_fn = jax.vmap(self._represent_trajectory, in_axes=(None, 0, 0, None, None, 0))
        latent_state_trajectory_batch, updated_memory_trajectory_batch = represent_trajectory_batch_fn(
            train_state.params,
            observation_trajectory_batch, memory_trajectory_batch,
            train_state, deterministic, represent_trajectory_key_batch)

        def prediction_fn(state, rng):
            log_action_probs, log_value_probs = train_state.prediction_fn(
                train_state.params, state, deterministic=deterministic, rngs={'dropout': rng})
            return log_action_probs, self._decode_value_or_reward(log_value_probs)

        def afterstate_prediction_fn(afterstate, rng):
            log_chance_outcome_probs, log_afterstate_value_probs = train_state.afterstate_prediction_fn(
                train_state.params, afterstate, deterministic=deterministic, rngs={'dropout': rng})
            return log_chance_outcome_probs, self._decode_value_or_reward(log_afterstate_value_probs)

        def dynamics_fn(afterstate, chance_outcome, rng):
            next_state, reward_log_probs = train_state.dynamics_fn(
                train_state.params, afterstate, chance_outcome, deterministic=deterministic, rngs={'dropout': rng})
            return next_state, self._decode_value_or_reward(reward_log_probs)

        def afterstate_dynamics_fn(prev_state, action, rng):
            afterstate = train_state.afterstate_dynamics_fn(
                train_state.params, prev_state, action, deterministic=deterministic, rngs={'dropout': rng})
            return afterstate

        mcts_func = functools.partial(
            mcts,
            num_actions=self.action_space.n,
            num_chance_outcomes=self._model.num_chance_outcomes,
            prediction_fn=prediction_fn,
            afterstate_prediction_fn=afterstate_prediction_fn,
            dynamics_fn=dynamics_fn,
            afterstate_dynamics_fn=afterstate_dynamics_fn,
            # The meaning of discount factor is different for Stochastic MuZero MCTS because
            # every transition is split into a transition to the afterstate and a transition to the next state.
            discount_factor=math.sqrt(self._config['discount_factor']),
            num_simulations=self._config['num_mcts_simulations'],
            puct_c1=self._config['mcts_puct_c1'],
            dirichlet_noise_alpha=self._config['mcts_dirichlet_noise_alpha'],
            root_exploration_fraction=self._config['mcts_root_exploration_fraction'],
            search_policy=self._config['mcts_search_policy'],
            result_policy=self._config['mcts_result_policy'],
        )
        trajectory_mcts = jax.vmap(mcts_func)
        trajectory_batch_mcts = jax.vmap(trajectory_mcts)
        mcts_key, rng = jax.random.split(rng)
        mcts_key_batch = jax.random.split(mcts_key, batch_size * num_timestamps).reshape(
            (batch_size, num_timestamps, 2))
        mcts_policy_log_probs, mcts_value, mcts_search_trees, mcts_stats = trajectory_batch_mcts(
            latent_state_trajectory_batch, mcts_key_batch)

        return updated_memory_trajectory_batch, mcts_policy_log_probs, mcts_value, mcts_stats

    @timeit
    @partial(jax.jit, static_argnums=(0,))
    def _act_on_batch_jit(self, observation_batch, memory_batch, train_state, rng):
        mcts_stats_key, action_key = jax.random.split(rng)

        # Add fake timestamp dim to make a rudimentary trajectory
        observation_trajectory_batch = pytree.expand_dims(observation_batch, axis=1)
        memory_trajectory_batch = pytree.expand_dims(memory_batch, axis=1)

        updated_memory, mcts_policy_log_probs, mcts_value, _ = self._compute_mcts_statistics(
            observation_trajectory_batch, memory_trajectory_batch,
            train_state, self._config['act_deterministic'], mcts_stats_key)

        # Get rid of the fake timestamp dimension that we've added before
        mcts_policy_log_probs = pytree.squeeze(mcts_policy_log_probs, axis=1)
        mcts_value = pytree.squeeze(mcts_value, axis=1)
        updated_memory = pytree.squeeze(updated_memory, axis=1)

        # Choose actions to execute in the environment
        selected_actions = jax.random.categorical(action_key, mcts_policy_log_probs)

        act_metadata = {
            'memory_state_after': updated_memory,
            'log_mcts_action_probs': mcts_policy_log_probs,
            'mcts_state_values': mcts_value,
        }

        return selected_actions, act_metadata

    @timeit
    def _train(self, training_batch):
        self._train_state, train_stats, per_trajectory_loss_details = self._train_jit(
            self._train_state, training_batch, self.next_random_key())
        train_stats = pytree.update(train_stats, self._replay_buffer.get_stats())
        return train_stats, per_trajectory_loss_details

    @timeit
    @partial(jax.jit, static_argnums=(0,))
    def _train_jit(self, train_state, training_batch, rng):
        def loss_function(params, rng):
            def trajectory_loss(params, trajectory, rng):
                deterministic = self._config['train_deterministic']

                num_unroll_steps = self._config['num_train_unroll_steps']

                # Convert observation trajectory into a sequence of latent states for each timestamp
                rng, representation_key = jax.random.split(rng)
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

                # Encode latent states with VQ-VAE chance outcome encoder
                def chance_outcome_encoder(state, rng):
                    return train_state.chance_outcome_encoder_fn(params, state, deterministic, rngs={'dropout': rng})
                batch_chance_outcome_encoder_fn = jax.vmap(chance_outcome_encoder)
                rng, chance_outcome_key = jax.random.split(rng)
                chance_outcome_key_batch = jax.random.split(chance_outcome_key, num_timestamps)
                encoded_chance_outcomes = batch_chance_outcome_encoder_fn(
                    trajectory_latent_states, chance_outcome_key_batch)
                num_chance_outcomes = encoded_chance_outcomes.shape[-1]
                chance_outcomes_one_hot = jax.nn.one_hot(
                    jnp.argmax(encoded_chance_outcomes, axis=-1), num_classes=num_chance_outcomes, dtype=jnp.float32)
                chance_outcomes_one_hot = encoded_chance_outcomes + jax.lax.stop_gradient(
                    chance_outcomes_one_hot - encoded_chance_outcomes)  # Straight-through estimator
                chance_outcome_commitment_loss = jnp.mean(
                    rlax.l2_loss(encoded_chance_outcomes, jax.lax.stop_gradient(chance_outcomes_one_hot))) / num_chance_outcomes
                # Pad so that we have valid values at the end of trajectory (loss will be masked there anyway)
                chance_outcomes_one_hot_padded = jnp.concatenate(
                    [
                        chance_outcomes_one_hot,
                        jax.nn.one_hot(
                            jnp.zeros(num_unroll_steps, dtype=jnp.int32),
                            num_classes=num_chance_outcomes,
                            dtype=jnp.float32
                        ),
                    ],
                    axis=0,
                )

                value_l2_loss = 0.0  # TODO: remove me
                value_loss = 0.0
                afterstate_value_loss = 0.0
                reward_loss = 0.0
                policy_loss = 0.0
                chance_outcome_prediction_loss = 0.0
                state_similarity_loss = 0.0
                state_avg_sqr = 0.0
                current_latent_states = trajectory_latent_states
                for unroll_step in range(num_unroll_steps):  # TODO: lax.fori ?
                    current_state_valid_mask = jnp.arange(0, num_timestamps) < num_timestamps - unroll_step
                    next_state_valid_mask = jnp.arange(num_timestamps) < num_timestamps - unroll_step - 1
                    current_state_valid_scale = 1.0 / jnp.sum(current_state_valid_mask)
                    next_state_valid_scale = 1.0 / jnp.sum(next_state_valid_mask)

                    actions = trajectory['training_targets']['actions'][:, unroll_step]
                    next_state_is_terminal_or_after = trajectory['training_targets']['next_state_is_terminal_or_after'][:, unroll_step]
                    next_state_is_terminal_or_after_extra_dim = jnp.expand_dims(
                        next_state_is_terminal_or_after, axis=1)
                    value_targets = trajectory['training_targets']['value_discrete'][:, unroll_step]
                    afterstate_value_targets = trajectory['training_targets']['afterstate_value_discrete'][:, unroll_step]
                    reward_targets = trajectory['training_targets']['rewards_discrete'][:, unroll_step]
                    policy_targets = trajectory['training_targets']['policy'][:, unroll_step]
                    state_targets = trajectory_latent_states_padded[unroll_step + 1: unroll_step + 1 + num_timestamps]
                    if self._config['state_similarity_loss_stop_gradient']:
                        state_targets = jax.lax.stop_gradient(state_targets)
                    chance_outcome_one_hot_targets = chance_outcomes_one_hot_padded[unroll_step + 1: unroll_step + 1 + num_timestamps]
                    # When there's no next state in trajectory due to termination,
                    # we use an arbitrary chance outcome (zero)
                    # TODO: it's incorrect to fix chance transition to the terminal state, do something!
                    chance_outcome_one_hot_targets = (
                        chance_outcome_one_hot_targets * (1 - next_state_is_terminal_or_after_extra_dim) +
                        jax.nn.one_hot(0, num_chance_outcomes, dtype=jnp.float32) * next_state_is_terminal_or_after_extra_dim
                    )

                    # Predict state value and the next action
                    def prediction_fn(state, rng):
                        return train_state.prediction_fn(params, state, deterministic, rngs={'dropout': rng})
                    rng, prediction_key = jax.random.split(rng)
                    prediction_key_batch = jax.random.split(prediction_key, num_timestamps)
                    batch_prediction_fn = jax.vmap(prediction_fn)
                    policy_log_probs, state_value_log_probs = batch_prediction_fn(current_latent_states, prediction_key_batch)

                    # Predict the afterstate
                    def afterstate_dynamics_fn(state, action, rng):
                        return train_state.afterstate_dynamics_fn(
                            params, state, action, deterministic, rngs={'dropout': rng})
                    rng, afterstate_dynamics_key = jax.random.split(rng)
                    afterstate_dynamics_key_batch = jax.random.split(afterstate_dynamics_key, num_timestamps)
                    batch_afterstate_dynamics_fn = jax.vmap(afterstate_dynamics_fn)
                    latent_afterstates = batch_afterstate_dynamics_fn(
                        current_latent_states, actions, afterstate_dynamics_key_batch)

                    # Predict the afterstate value and the chance outcome
                    def afterstate_prediction_fn(afterstate, rng):
                        return train_state.afterstate_prediction_fn(
                            params, afterstate, deterministic, rngs={'dropout': rng})
                    rng, afterstate_prediction_key = jax.random.split(rng)
                    afterstate_prediction_key_batch = jax.random.split(afterstate_prediction_key, num_timestamps)
                    batch_afterstate_prediction_fn = jax.vmap(afterstate_prediction_fn)
                    chance_outcome_log_probs, afterstate_value_log_probs = batch_afterstate_prediction_fn(
                        latent_afterstates, afterstate_prediction_key_batch)

                    # Predict the next state and the reward
                    def dynamics_fn(latent_afterstate, chance_outcome_one_hot, rng):
                        return train_state.dynamics_fn(
                            params, latent_afterstate, chance_outcome_one_hot, deterministic, rngs={'dropout': rng})
                    rng, dynamics_key = jax.random.split(rng)
                    dynamics_key_batch = jax.random.split(dynamics_key, num_timestamps)
                    batch_dynamics_fn = jax.vmap(dynamics_fn)
                    current_latent_states, reward_log_probs = batch_dynamics_fn(
                        latent_afterstates, chance_outcome_one_hot_targets, dynamics_key_batch)

                    # Compute prediction losses
                    step_value_loss = jax.vmap(rlax.categorical_cross_entropy)(value_targets, state_value_log_probs)
                    step_afterstate_value_loss = jax.vmap(rlax.categorical_cross_entropy)(
                        afterstate_value_targets, afterstate_value_log_probs)
                    step_reward_loss = jax.vmap(rlax.categorical_cross_entropy)(reward_targets, reward_log_probs)
                    step_policy_loss = jax.vmap(rlax.categorical_cross_entropy)(policy_targets, policy_log_probs)

                    # TODO: remove me, just for debug
                    value_scalar = self._decode_value_or_reward(state_value_log_probs)
                    value_target_scalar = trajectory['training_targets']['value_scalar'][:, unroll_step]
                    step_value_l2_loss = rlax.l2_loss(value_scalar, value_target_scalar)
                    value_l2_loss += jnp.sum(current_state_valid_mask * step_value_l2_loss) * current_state_valid_scale

                    # Mask out every timestamp for which we don't have a valid target
                    # Also apply loss scaling to make loss timestamp-independent

                    value_loss += jnp.sum(current_state_valid_mask * step_value_loss) * current_state_valid_scale
                    reward_loss += jnp.sum(current_state_valid_mask * step_reward_loss) * current_state_valid_scale
                    policy_loss += jnp.sum(current_state_valid_mask * step_policy_loss) * current_state_valid_scale
                    afterstate_value_loss += jnp.sum(
                        next_state_valid_mask * step_afterstate_value_loss) * next_state_valid_scale

                    # Compute chance outcome prediction loss
                    step_chance_outcome_prediction_loss = jax.vmap(rlax.categorical_cross_entropy)(
                        jax.lax.stop_gradient(chance_outcome_one_hot_targets), chance_outcome_log_probs)
                    chance_outcome_prediction_loss += jnp.sum(
                        next_state_valid_mask * step_chance_outcome_prediction_loss) * next_state_valid_scale

                    # Compute state similarity loss following "Improving Model-Based Reinforcement Learning
                    # with Internal State Representations through Self-Supervision",
                    # https://arxiv.org/abs/2102.05599
                    # TODO: use a BYOL-like method here
                    step_state_similarity_loss = jnp.mean(
                        rlax.l2_loss(current_latent_states, state_targets), axis=(-2, -1))
                    # We don't have targets at the end of a trajectory
                    # We also don't have next state targets for terminal states or anything after
                    state_similarity_loss_mask = next_state_valid_mask * (1.0 - next_state_is_terminal_or_after)
                    state_similarity_loss += (
                        jnp.sum(state_similarity_loss_mask * step_state_similarity_loss) /
                        (jnp.sum(state_similarity_loss_mask) + 1e-10)  # Just in case
                    )

                    state_avg_sqr += jnp.mean(rlax.l2_loss(current_latent_states))

                # Make loss independent of num_unroll_steps
                afterstate_value_loss /= num_unroll_steps
                value_loss /= num_unroll_steps
                value_l2_loss /= num_unroll_steps  # TODO: remove me
                reward_loss /= num_unroll_steps
                policy_loss /= num_unroll_steps
                chance_outcome_prediction_loss /= num_unroll_steps
                state_similarity_loss /= num_unroll_steps
                loss = (
                    self._config['afterstate_value_loss_weight'] * afterstate_value_loss +
                    self._config['value_loss_weight'] * value_loss +
                    self._config['reward_loss_weight'] * reward_loss +
                    self._config['policy_loss_weight'] * policy_loss +
                    self._config['chance_outcome_prediction_loss_weight'] * chance_outcome_prediction_loss +
                    self._config['chance_outcome_commitment_loss_weight'] * chance_outcome_commitment_loss +
                    self._config['state_similarity_loss_weight'] * state_similarity_loss
                )

                return loss, {
                    'afterstate_value_loss': afterstate_value_loss,
                    'value_loss': value_loss,
                    'value_l2_loss': value_l2_loss,  # TODO: remove em
                    'reward_loss': reward_loss,
                    'policy_loss': policy_loss,
                    'chance_outcome_prediction_loss': chance_outcome_prediction_loss,
                    'chance_outcome_commitment_loss': chance_outcome_commitment_loss,
                    'state_similarity_loss': state_similarity_loss,
                    'state_avg_sqr': state_avg_sqr,
                    'muzero_loss': loss,
                }

            batch_loss = jax.vmap(trajectory_loss, in_axes=(None, 0, 0))
            batch_size = pytree.get_axis_dim(training_batch, axis=0)
            loss_key = jax.random.split(rng, batch_size)
            per_trajectory_losses, per_trajectory_loss_details = batch_loss(params, training_batch, loss_key)
            loss = jnp.mean(per_trajectory_losses)

            return loss, per_trajectory_loss_details

        grad_and_loss_details_func = jax.grad(loss_function, argnums=0, has_aux=True)
        grads, per_trajectory_loss_details = grad_and_loss_details_func(train_state.params, rng)
        train_state = train_state.apply_gradients(grads=grads)

        stats = pytree.mean(per_trajectory_loss_details)

        return train_state, stats, per_trajectory_loss_details

    # TODO: for some reason total execution time of this function
    # TODO: is 200 ms more than _make_next_training_batch_jit + sample_trajectory_batch
    @timeit
    def _make_next_training_batch(self):
        trajectory_batch_items = self._replay_buffer.sample_trajectory_batch(self._config['reanalyze_batch_size'])
        # Get rid of CPU-side copies of tensors before batching
        trajectories = [
            pytree.remove_keys(item.trajectory, ['rewards_cpu'])
            for item in trajectory_batch_items
        ]
        trajectory_batch = jax.jit(pytree.stack, static_argnames=['axis'])(trajectories, axis=0)

        training_batch, reanalyse_stats = self._make_next_training_batch_jit(
            self._train_state, trajectory_batch, self.next_random_key())

        return training_batch, reanalyse_stats, trajectory_batch_items

    @timeit
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
            'reanalyze_avg_mcts_policy_entropy': jnp.mean(
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

            # Allocate memory for training targets.
            # We will pre-compute training targets for every unroll step where possible.
            trajectory = pytree.update(trajectory, {
                'training_targets': {
                    'value_scalar': jnp.zeros((num_timestamps, num_unroll_steps), dtype=jnp.float32),
                    'afterstate_value_scalar': jnp.zeros((num_timestamps, num_unroll_steps), dtype=jnp.float32),
                    'rewards_scalar': jnp.zeros((num_timestamps, num_unroll_steps), dtype=jnp.float32),
                    'policy': jnp.zeros((num_timestamps, num_unroll_steps, num_actions), dtype=jnp.float32),
                    'actions': jnp.zeros((num_timestamps, num_unroll_steps), dtype=jnp.int32),
                    'next_state_is_terminal_or_after': jnp.zeros((num_timestamps, num_unroll_steps), dtype=jnp.bool_),
                }
            })

            # Padding will make computations at the end of a trajectory more convenient
            assert num_timestamps >= num_unroll_steps, \
                'There will be unroll steps with no ground truth at all otherwise'
            padded_target_sources = {
                'state_values': jnp.concatenate(
                    # We need an extra padded value here to compute afterstate values
                    [state_values, jnp.zeros(num_unroll_steps, dtype=jnp.float32)]
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

                actions = jax.lax.dynamic_slice_in_dim(
                    padded_target_sources['actions'], unroll_step, num_timestamps, axis=0)
                # Value and reward targets are zero after we've reached a terminal state
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

                afterstate_value_targets = (
                    reward_targets_scalar +
                    # We take square root here for consistency with interpretation of discount factor in MCTS.
                    math.sqrt(self._config['discount_factor']) * jax.lax.dynamic_slice_in_dim(
                        padded_target_sources['state_values'], unroll_step + 1, num_timestamps, axis=0) *
                    (1.0 - next_state_is_terminal_or_after)
                )

                trajectory['training_targets'] = {
                    'value_scalar': trajectory['training_targets']['value_scalar'].at[:, unroll_step].set(value_targets),
                    'afterstate_value_scalar': trajectory['training_targets']['afterstate_value_scalar'].at[:, unroll_step].set(
                        afterstate_value_targets),
                    'rewards_scalar': trajectory['training_targets']['rewards_scalar'].at[:, unroll_step].set(
                        reward_targets_scalar),
                    'policy': trajectory['training_targets']['policy'].at[:, unroll_step].set(policy_target_probs),
                    'actions': trajectory['training_targets']['actions'].at[:, unroll_step].set(
                        actions),
                    'next_state_is_terminal_or_after':
                        trajectory['training_targets']['next_state_is_terminal_or_after'].at[:, unroll_step].set(
                            next_state_is_terminal_or_after),
                }

                return trajectory, padded_target_sources, next_state_is_terminal_or_after

            trajectory_with_targets, _, _ = jax.lax.fori_loop(
                lower=0, upper=num_unroll_steps, body_fun=unroll_loop_body,
                init_val=(trajectory, padded_target_sources, jnp.zeros_like(trajectory['done']))
            )

            # Convert values and rewards to discrete representation
            trajectory_with_targets['training_targets']['rewards_discrete'] = self._encode_value_or_reward(
                trajectory_with_targets['training_targets']['rewards_scalar'])
            trajectory_with_targets['training_targets']['value_discrete'] = self._encode_value_or_reward(
                trajectory_with_targets['training_targets']['value_scalar'])
            trajectory_with_targets['training_targets']['afterstate_value_discrete'] = self._encode_value_or_reward(
                trajectory_with_targets['training_targets']['afterstate_value_scalar'])

            return trajectory_with_targets

        return jax.vmap(make_training_trajectory)(trajectory_batch)
