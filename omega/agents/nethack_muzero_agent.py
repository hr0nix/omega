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
import jax.experimental.checkify as checkify
import rlax
import optax
import optax_adan

from ..math.probability import entropy, cross_entropy
from ..utils import pytree
from ..utils.flax import merge_params
from ..utils.jax import throws_on_checkify_error, checkify_method
from ..utils.profiling import timeit
from ..utils.tensor import pad_zeros, pad_values, pad_one_hot, masked_mean
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
        'mcts_value_reward_ensemble_size': 1,
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

        self._value_reward_transform_pair = make_value_reward_transform_pair(
            min_value=self._config['value_reward_min_max'][0],
            max_value=self._config['value_reward_min_max'][1],
            num_bins=self._config['value_reward_bins'],
        )

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
            #optax.adamw(learning_rate=lr_schedule, weight_decay=self._config['weight_decay']),
            optax_adan.adan(learning_rate=lr_schedule),
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

    @throws_on_checkify_error
    @partial(jax.jit, static_argnums=0)
    @checkify_method
    def _get_current_batch_stats_jit(self, trajectory_batch):
        return {
            'act_avg_mcts_policy_entropy': jnp.mean(
                entropy(logits=trajectory_batch['act_metadata']['log_mcts_action_probs'])
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
    @throws_on_checkify_error
    @partial(jax.jit, static_argnums=0)
    @checkify_method
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
    @throws_on_checkify_error
    @partial(jax.jit, static_argnums=0)
    @checkify_method
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
    @throws_on_checkify_error
    @partial(jax.jit, static_argnums=0)
    @checkify_method
    def _update_initial_trajectory_memory_jit(self, memories_to_update, update_source_ids, update_source):
        def updater(memory_to_update, update_source_id):
            update = update_source[update_source_id]
            memory_abs_diff = jnp.mean(jnp.abs(memory_to_update[0] - update))
            updated_memory = memory_to_update.at[0].set(update)
            return updated_memory, memory_abs_diff
        updated = jax.tree_util.tree_map(updater, memories_to_update, update_source_ids)
        updated_memories, memory_abs_diffs = jax.tree_util.tree_transpose(
            inner_treedef=jax.tree_util.tree_structure([_ for _ in range(2)]),
            outer_treedef=jax.tree_util.tree_structure(memories_to_update),
            pytree_to_transpose=updated
        )
        return updated_memories, pytree.array_mean(memory_abs_diffs)

    @timeit
    def _update_next_trajectory_memory(self, replayed_items, training_batch):
        """
        Given that we have an updated memory for each trajectory in the batch,
        we can update initial memories of the trajectories that temporally follow the batch
        (provided that they still are present in the replay buffer).
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
                # has been evicted from the buffer (this can happen
                # with some replay buffer types, i.e. clustering replay)
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
        batch_size, num_timestamps = jax.tree_util.tree_leaves(observation_trajectory_batch)[0].shape[:2]

        rng, represent_trajectory_key = jax.random.split(rng)
        represent_trajectory_key_batch = jax.random.split(represent_trajectory_key, batch_size)
        represent_trajectory_batch_fn = jax.vmap(self._represent_trajectory, in_axes=(None, 0, 0, None, None, 0))
        latent_state_trajectory_batch, updated_memory_trajectory_batch = represent_trajectory_batch_fn(
            train_state.params,
            observation_trajectory_batch, memory_trajectory_batch,
            train_state, deterministic, represent_trajectory_key_batch)

        def prediction_fn(state, rng):
            log_action_probs, log_value_probs = train_state.prediction_fn(
                train_state.params, state,
                ensemble_size=self._config['mcts_value_reward_ensemble_size'],
                deterministic=deterministic, rngs={'dropout': rng})
            value_probs = jnp.exp(log_value_probs)
            return log_action_probs, self._value_reward_transform_pair.apply_inv(value_probs)

        def afterstate_prediction_fn(afterstate, rng):
            log_chance_outcome_probs, log_afterstate_value_probs = train_state.afterstate_prediction_fn(
                train_state.params, afterstate,
                ensemble_size=self._config['mcts_value_reward_ensemble_size'],
                deterministic=deterministic, rngs={'dropout': rng})
            afterstate_value_probs = jnp.exp(log_afterstate_value_probs)
            return log_chance_outcome_probs, self._value_reward_transform_pair.apply_inv(afterstate_value_probs)

        def dynamics_fn(afterstate, chance_outcome, rng):
            next_state, log_reward_probs = train_state.dynamics_fn(
                train_state.params, afterstate, chance_outcome,
                ensemble_size=self._config['mcts_value_reward_ensemble_size'],
                deterministic=deterministic, rngs={'dropout': rng})
            reward_probs = jnp.exp(log_reward_probs)
            return next_state, self._value_reward_transform_pair.apply_inv(reward_probs)

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
    @partial(jax.jit, static_argnums=0)  # TODO: enable full checkify
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
    @throws_on_checkify_error
    @partial(jax.jit, static_argnums=0)
    @checkify_method
    def _train_jit(self, train_state, training_batch, rng):
        """
        Updates the model parameters by training on the given training batch.
        """
        def loss_function(params, rng):
            def trajectory_loss(params, trajectory, rng):
                deterministic = self._config['train_deterministic']

                num_unroll_steps = self._config['num_train_unroll_steps']

                # Convert observation trajectory into a sequence of latent states for each timestamp
                rng, representation_key = jax.random.split(rng)
                trajectory_latent_states, _ = self._represent_trajectory(
                    params, trajectory['current_state'], trajectory['memory_before'],
                    train_state, deterministic, representation_key)
                trajectory_latent_states_padded = pad_zeros(trajectory_latent_states, num_unroll_steps, axis=0)
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
                    rlax.l2_loss(encoded_chance_outcomes, jax.lax.stop_gradient(chance_outcomes_one_hot)))
                # Pad so that we have valid values at the end of trajectory (loss will be masked there anyway)
                chance_outcomes_one_hot_padded = pad_one_hot(chance_outcomes_one_hot, num_unroll_steps, 0)

                # TODO: remove me
                afterstate_value_l2_loss = 0.0
                state_value_l2_loss = 0.0
                reward_l2_loss = 0.0
                # end of TODO
                state_value_loss = 0.0
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

                    actions = trajectory['training_targets']['action'][:, unroll_step]
                    next_state_is_terminal_or_after = trajectory['training_targets']['next_state_is_terminal_or_after'][:, unroll_step]
                    next_state_is_terminal_or_after_extra_dim = jnp.expand_dims(
                        next_state_is_terminal_or_after, axis=1)
                    state_value_targets = trajectory['training_targets']['state_value_discrete'][:, unroll_step]
                    afterstate_value_targets = trajectory['training_targets']['afterstate_value_discrete'][:, unroll_step]
                    reward_targets = trajectory['training_targets']['reward_discrete'][:, unroll_step]
                    policy_targets = trajectory['training_targets']['policy'][:, unroll_step]
                    state_targets = trajectory_latent_states_padded[unroll_step + 1: unroll_step + 1 + num_timestamps]
                    if self._config['state_similarity_loss_stop_gradient']:
                        state_targets = jax.lax.stop_gradient(state_targets)
                    chance_outcome_one_hot_targets = chance_outcomes_one_hot_padded[unroll_step + 1: unroll_step + 1 + num_timestamps]
                    # When there's no next state in trajectory due to termination,
                    # we use an arbitrary chance outcome (zero)
                    # TODO: it's incorrect to fix chance transition to the terminal state, find a way to fix it!
                    chance_outcome_one_hot_targets = jnp.where(
                        next_state_is_terminal_or_after_extra_dim,
                        jax.nn.one_hot(0, num_chance_outcomes, dtype=jnp.float32),
                        chance_outcome_one_hot_targets,
                    )

                    # Predict state value and the next action
                    def prediction_fn(state, rng):
                        return train_state.prediction_fn(
                            params, state, deterministic=deterministic, rngs={'dropout': rng})
                    rng, prediction_key = jax.random.split(rng)
                    prediction_key_batch = jax.random.split(prediction_key, num_timestamps)
                    batch_prediction_fn = jax.vmap(prediction_fn)
                    policy_log_probs, state_value_log_probs = batch_prediction_fn(
                        current_latent_states, prediction_key_batch)

                    # Predict the afterstate
                    def afterstate_dynamics_fn(state, action, rng):
                        return train_state.afterstate_dynamics_fn(
                            params, state, action, deterministic=deterministic, rngs={'dropout': rng})
                    rng, afterstate_dynamics_key = jax.random.split(rng)
                    afterstate_dynamics_key_batch = jax.random.split(afterstate_dynamics_key, num_timestamps)
                    batch_afterstate_dynamics_fn = jax.vmap(afterstate_dynamics_fn)
                    latent_afterstates = batch_afterstate_dynamics_fn(
                        current_latent_states, actions, afterstate_dynamics_key_batch)

                    # Predict the afterstate value and the chance outcome
                    def afterstate_prediction_fn(afterstate, rng):
                        return train_state.afterstate_prediction_fn(
                            params, afterstate, deterministic=deterministic, rngs={'dropout': rng})
                    rng, afterstate_prediction_key = jax.random.split(rng)
                    afterstate_prediction_key_batch = jax.random.split(afterstate_prediction_key, num_timestamps)
                    batch_afterstate_prediction_fn = jax.vmap(afterstate_prediction_fn)
                    chance_outcome_log_probs, afterstate_value_log_probs = batch_afterstate_prediction_fn(
                        latent_afterstates, afterstate_prediction_key_batch)

                    # Predict the next state and the reward
                    def dynamics_fn(latent_afterstate, chance_outcome_one_hot, rng):
                        return train_state.dynamics_fn(
                            params, latent_afterstate, chance_outcome_one_hot,
                            deterministic=deterministic, rngs={'dropout': rng})
                    rng, dynamics_key = jax.random.split(rng)
                    dynamics_key_batch = jax.random.split(dynamics_key, num_timestamps)
                    batch_dynamics_fn = jax.vmap(dynamics_fn)
                    current_latent_states, reward_log_probs = batch_dynamics_fn(
                        latent_afterstates, chance_outcome_one_hot_targets, dynamics_key_batch)

                    # Compute prediction losses
                    step_state_value_loss = cross_entropy(labels=state_value_targets, logits=state_value_log_probs)
                    step_afterstate_value_loss = cross_entropy(
                        labels=afterstate_value_targets, logits=afterstate_value_log_probs)
                    step_reward_loss = cross_entropy(labels=reward_targets, logits=reward_log_probs)
                    step_policy_loss = cross_entropy(labels=policy_targets, logits=policy_log_probs)

                    # TODO: remove me, just for debug
                    afterstate_value_probs = jnp.exp(afterstate_value_log_probs)
                    state_value_probs = jnp.exp(state_value_log_probs)
                    reward_probs = jnp.exp(reward_log_probs)
                    afterstate_value_scalar = self._value_reward_transform_pair.apply_inv(afterstate_value_probs)
                    state_value_scalar = self._value_reward_transform_pair.apply_inv(state_value_probs)
                    reward_scalar = self._value_reward_transform_pair.apply_inv(reward_probs)
                    afterstate_target_scalar = trajectory['training_targets']['afterstate_value_scalar'][:, unroll_step]
                    state_value_target_scalar = trajectory['training_targets']['state_value_scalar'][:, unroll_step]
                    reward_target_scalar = trajectory['training_targets']['reward_scalar'][:, unroll_step]
                    step_afterstate_value_l2_loss = rlax.l2_loss(afterstate_value_scalar, afterstate_target_scalar)
                    step_state_value_l2_loss = rlax.l2_loss(state_value_scalar, state_value_target_scalar)
                    step_reward_l2_loss = rlax.l2_loss(reward_scalar, reward_target_scalar)
                    afterstate_value_l2_loss += masked_mean(step_afterstate_value_l2_loss, next_state_valid_mask)
                    state_value_l2_loss += masked_mean(step_state_value_l2_loss, current_state_valid_mask)
                    reward_l2_loss += masked_mean(step_reward_l2_loss, current_state_valid_mask)

                    # Mask out every timestamp for which we don't have a valid target
                    # Also apply loss scaling to make loss timestamp-independent

                    state_value_loss += masked_mean(step_state_value_loss, current_state_valid_mask)
                    reward_loss += masked_mean(step_reward_loss, current_state_valid_mask)
                    policy_loss += masked_mean(step_policy_loss, current_state_valid_mask)
                    afterstate_value_loss += masked_mean(step_afterstate_value_loss, next_state_valid_mask)

                    # Compute chance outcome prediction loss
                    step_chance_outcome_prediction_loss = cross_entropy(
                        jax.lax.stop_gradient(chance_outcome_one_hot_targets), chance_outcome_log_probs)
                    chance_outcome_prediction_loss += masked_mean(
                        step_chance_outcome_prediction_loss, next_state_valid_mask)

                    # Compute state similarity loss following "Improving Model-Based Reinforcement Learning
                    # with Internal State Representations through Self-Supervision",
                    # https://arxiv.org/abs/2102.05599
                    # TODO: use a BYOL-like method here
                    step_state_similarity_loss = jnp.mean(
                        rlax.l2_loss(current_latent_states, state_targets), axis=(-2, -1))
                    # We don't have targets at the end of a trajectory
                    # We also don't have next state targets for terminal states or anything after
                    state_similarity_loss_mask = jnp.logical_and(
                        next_state_valid_mask, jnp.logical_not(next_state_is_terminal_or_after))
                    state_similarity_loss += masked_mean(
                        step_state_similarity_loss, state_similarity_loss_mask, allow_zero_mask=True)

                    # Monitor the magnitude of the states to detect divergence
                    state_avg_sqr += jnp.mean(rlax.l2_loss(current_latent_states))

                    # TODO: might remove this after debugging
                    checkify.check(state_value_loss != jnp.inf, 'value loss is inf')
                    checkify.check(afterstate_value_loss != jnp.inf, 'afterstate value loss is inf')
                    checkify.check(reward_loss != jnp.inf, 'reward loss is inf')
                    checkify.check(policy_loss != jnp.inf, 'policy loss is inf')
                    checkify.check(state_similarity_loss != jnp.inf, 'state similarity loss is inf')
                    checkify.check(chance_outcome_prediction_loss != jnp.inf, 'chance prediction loss is inf')

                # Make loss independent of num_unroll_steps
                afterstate_value_loss /= num_unroll_steps
                state_value_loss /= num_unroll_steps
                reward_loss /= num_unroll_steps
                policy_loss /= num_unroll_steps
                chance_outcome_prediction_loss /= num_unroll_steps
                state_similarity_loss /= num_unroll_steps
                state_avg_sqr /= num_unroll_steps
                # TODO: remove me
                afterstate_value_l2_loss /= num_unroll_steps
                state_value_l2_loss /= num_unroll_steps
                reward_l2_loss /= num_unroll_steps
                # end of TODO
                loss = (
                    self._config['afterstate_value_loss_weight'] * afterstate_value_loss +
                    self._config['value_loss_weight'] * state_value_loss +
                    self._config['reward_loss_weight'] * reward_loss +
                    self._config['policy_loss_weight'] * policy_loss +
                    self._config['chance_outcome_prediction_loss_weight'] * chance_outcome_prediction_loss +
                    self._config['chance_outcome_commitment_loss_weight'] * chance_outcome_commitment_loss +
                    self._config['state_similarity_loss_weight'] * state_similarity_loss
                )

                return loss, {
                    'afterstate_value_loss': afterstate_value_loss,
                    'value_loss': state_value_loss,
                    'reward_loss': reward_loss,
                    # TODO: remove me
                    'afterstate_value_l2_loss': afterstate_value_l2_loss,
                    'value_l2_loss': state_value_l2_loss,
                    'reward_l2_loss': reward_l2_loss,
                    # end of TODO
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

    @timeit
    def _make_next_training_batch(self):
        """
        Samples a new batch of trajectories from the replay buffer, reanalyzes it and pre-computes training targets.
        """
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
    @throws_on_checkify_error
    @partial(jax.jit, static_argnums=0)  # TODO: enable full checkify
    def _make_next_training_batch_jit(self, train_state, trajectory_batch, rng):
        """
        Reanalyzes the trajectories in the training batch and pre-computes training targets based
        on the results of the analysis.
        """
        trajectory_batch_with_mcts_stats, reanalyze_stats = self._reanalyze(train_state, trajectory_batch, rng)
        # TODO: remove err-related stuff after checkify is enabled globally
        err, trajectory_batch_with_training_targets = self._prepare_training_targets(trajectory_batch_with_mcts_stats)
        return err, (trajectory_batch_with_training_targets, reanalyze_stats)

    def _reanalyze(self, train_state, trajectory_batch, rng):
        """
        Reanalyzes the trajectories using the current version of the model and augments training batch with MCTS stats.
        """
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
                entropy(logits=trajectory_batch['mcts_reanalyze']['log_action_probs'])
            ),
            'reanalyze_avg_mcts_state_value': jnp.mean(trajectory_batch['mcts_reanalyze']['state_values']),
            'reanalyze_var_mcts_state_value': jnp.var(trajectory_batch['mcts_reanalyze']['state_values']),
        }
        reanalyze_stats = pytree.update(reanalyze_stats, pytree.mean(mcts_stats))
        return trajectory_batch, reanalyze_stats

    @checkify_method
    def _prepare_training_targets(self, trajectory_batch):
        """
        Pre-computes training targets for all unroll steps and adds them to the training batch.
        """
        def make_training_trajectory(trajectory):
            targets = compute_training_targets(
                trajectory, self._config['num_train_unroll_steps'], self._config['discount_factor'])
            targets = pytree.update(targets, compute_one_hot_targets(targets, self._value_reward_transform_pair.apply))
            return pytree.update(trajectory, {
                'training_targets': targets,
            })

        return jax.vmap(make_training_trajectory)(trajectory_batch)


def compute_training_targets(trajectory, num_unroll_steps, discount_factor):
    trajectory_elements = [
        trajectory['mcts_reanalyze']['state_values'],
        trajectory['mcts_reanalyze']['log_action_probs'],
        trajectory['actions'],
        trajectory['rewards'],
        trajectory['done'],
    ]
    chex.assert_equal_shape_prefix(trajectory_elements, 1)
    chex.assert_type(trajectory_elements, [jnp.float32, jnp.float32, jnp.int32, jnp.float32, jnp.bool_])
    chex.assert_rank(trajectory_elements, [1, 2, 1, 1, 1])

    num_timestamps, num_actions = trajectory['mcts_reanalyze']['log_action_probs'].shape

    # Allocate memory for training targets. We will pre-compute training targets for every unroll step
    # except those that cannot be computed outside the training loop.
    trajectory_targets = {
        'state_value_scalar': jnp.zeros((num_timestamps, num_unroll_steps), dtype=jnp.float32),
        'afterstate_value_scalar': jnp.zeros((num_timestamps, num_unroll_steps), dtype=jnp.float32),
        'reward_scalar': jnp.zeros((num_timestamps, num_unroll_steps), dtype=jnp.float32),
        'policy': jnp.zeros((num_timestamps, num_unroll_steps, num_actions), dtype=jnp.float32),
        'action': jnp.zeros((num_timestamps, num_unroll_steps), dtype=jnp.int32),
        'next_state_is_terminal_or_after': jnp.zeros((num_timestamps, num_unroll_steps), dtype=jnp.bool_),
    }

    # Padding will make computations at the end of a trajectory more convenient
    assert num_timestamps >= num_unroll_steps, \
        'There will be unroll steps with no ground truth at all otherwise'
    # The original MuZero paper used n-step returns, but we'll go completely off-policy
    # by predicting the MCTS policy value.
    # MuZero Reanalyze paper has shown that it works, but a bit worse for some reason. Works fine for us.
    # We pad for num_unroll_steps here because we need an extra padded value to compute afterstate values.
    value_target_source = pad_zeros(trajectory['mcts_reanalyze']['state_values'], size=num_unroll_steps, axis=-1)
    actions_target_source = pad_zeros(trajectory['actions'], size=num_unroll_steps - 1, axis=-1)
    done_target_source = pad_zeros(trajectory['done'], size=num_unroll_steps - 1, axis=-1)
    reward_target_source = pad_zeros(trajectory['rewards'], size=num_unroll_steps - 1, axis=-1)
    # Pad action probs with uniform policy
    policy_target_source = pad_values(
        jnp.exp(trajectory['mcts_reanalyze']['log_action_probs']),
        size=num_unroll_steps - 1, value=1.0 / num_actions, axis=-2)

    def unroll_step_loop_body(unroll_step, loop_state):
        trajectory_targets, state_is_terminal_or_after = loop_state

        state_is_terminal_or_after_extra_dim = jnp.expand_dims(state_is_terminal_or_after, axis=-1)

        actions = jax.lax.dynamic_slice_in_dim(actions_target_source, unroll_step, num_timestamps, axis=0)
        state_value_targets_scalar = jax.lax.dynamic_slice_in_dim(value_target_source, unroll_step, num_timestamps, axis=0)
        reward_targets_scalar = jax.lax.dynamic_slice_in_dim(reward_target_source, unroll_step, num_timestamps, axis=0)
        policy_targets = jax.lax.dynamic_slice_in_dim(policy_target_source, unroll_step, num_timestamps, axis=0)

        # Value and reward targets are zero after we've reached a terminal state
        state_value_targets_scalar = jnp.where(state_is_terminal_or_after, 0.0, state_value_targets_scalar)
        reward_targets_scalar = jnp.where(state_is_terminal_or_after, 0.0, reward_targets_scalar)

        # Policy target is uniform after we've reached a terminal state
        policy_targets = jnp.where(
            state_is_terminal_or_after_extra_dim, jnp.full_like(policy_targets, 1.0 / num_actions), policy_targets)

        done = jax.lax.dynamic_slice_in_dim(done_target_source, unroll_step, num_timestamps, axis=0)
        next_state_is_terminal_or_after = jnp.logical_or(state_is_terminal_or_after, done)

        next_state_value_targets_scalar = jax.lax.dynamic_slice_in_dim(
            value_target_source, unroll_step + 1, num_timestamps, axis=0)
        afterstate_value_targets_scalar = (
            reward_targets_scalar +
            math.sqrt(discount_factor) *  # We take square root here for consistency with discount factor passed to MCTS
            jnp.where(next_state_is_terminal_or_after, 0.0, next_state_value_targets_scalar)
        )

        # Update the current slice of the training targets with the computed values
        trajectory_targets = pytree.update(trajectory_targets, {
            'state_value_scalar': trajectory_targets['state_value_scalar'].at[:, unroll_step].set(
                state_value_targets_scalar),
            'afterstate_value_scalar': trajectory_targets['afterstate_value_scalar'].at[:, unroll_step].set(
                afterstate_value_targets_scalar),
            'reward_scalar': trajectory_targets['reward_scalar'].at[:, unroll_step].set(reward_targets_scalar),
            'policy': trajectory_targets['policy'].at[:, unroll_step].set(policy_targets),
            'action': trajectory_targets['action'].at[:, unroll_step].set(actions),
            'next_state_is_terminal_or_after':
                trajectory_targets['next_state_is_terminal_or_after'].at[:, unroll_step].set(
                    next_state_is_terminal_or_after),
        })

        return trajectory_targets, next_state_is_terminal_or_after

    trajectory_targets, _ = jax.lax.fori_loop(
        lower=0, upper=num_unroll_steps, body_fun=unroll_step_loop_body,
        init_val=(trajectory_targets, jnp.zeros_like(trajectory['done']))
    )

    return trajectory_targets


def compute_one_hot_targets(trajectory_targets, value_reward_transform):
    return {
        'reward_discrete': value_reward_transform(trajectory_targets['reward_scalar']),
        'state_value_discrete': value_reward_transform(trajectory_targets['state_value_scalar']),
        'afterstate_value_discrete': value_reward_transform(trajectory_targets['afterstate_value_scalar']),
    }


def make_value_reward_transform_pair(min_value, max_value, num_bins):
    def apply(value):
        probs = rlax.transform_to_2hot(value, min_value, max_value, num_bins)
        chex.assert_rank(probs, len(value.shape) + 1)
        return probs

    def apply_inv(probs):
        chex.assert_axis_dimension(probs, axis=-1, expected=num_bins)
        result = rlax.transform_from_2hot(probs, min_value, max_value, num_bins)
        if len(probs.shape) == 1:
            # Work around a weird behavior of rlax.transform_from_2hot which doesn't return scalar outputs
            chex.assert_rank(result, 1)
            chex.assert_axis_dimension(result, 0, 1)
            result = jnp.squeeze(result, axis=-1)
        chex.assert_rank(result, len(probs.shape) - 1)
        return result

    return rlax.TxPair(apply=apply, apply_inv=apply_inv)
