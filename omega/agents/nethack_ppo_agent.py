import os

from functools import partial

from absl import logging

import flax.training.train_state
import flax.training.checkpoints
import jax.numpy as jnp
import jax.random
import optax
import rlax
from rlax._src import distributions

from ..utils import pytree
from .trainable_agent import JaxTrainableAgentBase
from ..neural.optimization import clip_gradient_by_norm
from ..models import NethackRNDNetworkPair, NethackPerceiverActorCriticModel


class NethackPPOAgent(JaxTrainableAgentBase):
    CONFIG = flax.core.frozen_dict.FrozenDict({
        'value_function_loss_weight': 1.0,
        'entropy_regularizer_weight': 0.0,
        'inverse_dynamics_loss_weight': 0.0,
        'ppo_loss_weight': 1.0,
        'lr': 1e-3,
        'discount_factor': 0.99,
        'gae_lambda': 0.95,
        'gradient_clipnorm': None,
        'num_minibatches_per_train_step': 100,
        'minibatch_size': 64,
        'ppo_eps': 0.25,
        'model_config': {},
        'use_rnd': False,
        'rnd_lr': 1e-3,
        'rnd_model_config': {},
        'exploration_reward_scale': 1.0,
    })

    class TrainState(flax.training.train_state.TrainState):
        step_index: int = None

    def __init__(self, *args, model_factory=NethackPerceiverActorCriticModel, config=None, **kwargs):
        super(NethackPPOAgent, self).__init__(*args, **kwargs)

        self._config = self.CONFIG.copy(config or {})
        self._model_factory = model_factory

        current_state_batch = self._make_fake_state_batch()
        self._model, self._train_state = self._build_model(current_state_batch)
        if self._config['use_rnd']:
            self._rnd_model, self._rnd_train_state = self._build_rnd_model(current_state_batch)
        else:
            self._rnd_model = self._rnd_train_state = None

    def _make_fake_state_batch(self):
        return {
            key: jnp.zeros(
                shape=(1,) + desc.shape,
                dtype=desc.dtype
            )
            for key, desc in self.observation_space.spaces.items()
        }

    def _build_model(self, state_batch):
        model = self._model_factory(num_actions=self.action_space.n, **self._config['model_config'])
        model_params = model.init(
            self.next_random_key(), current_state=state_batch, next_state=state_batch,
            deterministic=False, rng=self.next_random_key())

        optimizer = optax.adam(learning_rate=self._config['lr'])

        train_state = self.TrainState.create(apply_fn=model.apply, params=model_params, tx=optimizer)

        return model, train_state

    def _build_rnd_model(self, state_batch):
        rnd_model = NethackRNDNetworkPair(**self._config['rnd_model_config'])
        rnd_model_params = rnd_model.init(
            self.next_random_key(), state_batch, deterministic=False, rng=self.next_random_key())

        optimizer = optax.adam(learning_rate=self._config['rnd_lr'])

        rnd_train_state = self.TrainState.create(apply_fn=rnd_model.apply, params=rnd_model_params, tx=optimizer)

        return rnd_model, rnd_train_state

    def _rnd_checkpoint_path(self, checkpoint_path):
        return os.path.join(checkpoint_path, 'rnd')

    def try_load_from_checkpoint(self, path):
        """
        Loads a checkpoint from the given path if there are any.

        :param path: The path to checkpoints.
        :return: The index of the step that should be next.
        """
        checkpoint_path = flax.training.checkpoints.latest_checkpoint(path)
        if checkpoint_path is None:
            logging.warning('No checkpoints available at {}'.format(path))
            return 0
        else:
            logging.info('State will be loaded from checkpoint {}'.format(checkpoint_path))
            self._train_state = flax.training.checkpoints.restore_checkpoint(checkpoint_path, self._train_state)

            if self._config['use_rnd']:
                rnd_path = self._rnd_checkpoint_path(path)
                rnd_checkpoint_path = flax.training.checkpoints.latest_checkpoint(rnd_path)
                assert rnd_checkpoint_path is not None, f'No RND checkpoint found at {rnd_path}'
                self._rnd_train_state = flax.training.checkpoints.restore_checkpoint(
                    rnd_checkpoint_path, self._rnd_train_state)

            return self._train_state.step_index + 1

    def save_to_checkpoint(self, checkpoint_path, step):
        self._train_state = self._train_state.replace(step_index=step)
        flax.training.checkpoints.save_checkpoint(
            checkpoint_path, self._train_state, step=step, keep=1, overwrite=True)

        if self._config['use_rnd']:
            flax.training.checkpoints.save_checkpoint(
                self._rnd_checkpoint_path(checkpoint_path), self._rnd_train_state, step=step, keep=1, overwrite=True)

    @partial(jax.jit, static_argnums=(0,))
    def _act_on_batch_jitted(self, train_state, rnd_train_state, observation_batch, rng):
        rng, subkey1, subkey2 = jax.random.split(rng, 3)
        log_action_probs, _, state_values = train_state.apply_fn(
            train_state.params,
            current_state=observation_batch,
            next_state=observation_batch,  # Pass current state as the next state, inverse dynamics is irrelevant here
            deterministic=True, rng=subkey1)
        metadata = {
            'log_action_probs': log_action_probs,
            'state_values': state_values,
        }
        selected_actions = jax.random.categorical(subkey2, log_action_probs)

        if self._config['use_rnd']:
            rng, subkey = jax.random.split(rng)
            rnd_loss = rnd_train_state.apply_fn(
                rnd_train_state.params, observation_batch, deterministic=True, rng=subkey)
            metadata = pytree.update(metadata, {
                'rnd_loss': rnd_loss,
            })

        return selected_actions, metadata

    def _train_on_minibatch(self, train_state, rnd_train_state, trajectory_minibatch, rng):
        subkey1, subkey2 = jax.random.split(rng)

        def loss_function(params, train_state, trajectory_minibatch, rng):
            log_action_probs, log_id_action_probs, state_values = train_state.apply_fn(
                params, trajectory_minibatch['current_state'], trajectory_minibatch['next_state'],
                deterministic=False, rng=rng)

            minibatch_range = jnp.arange(0, log_action_probs.shape[0])

            log_sampled_action_probs = log_action_probs[minibatch_range, trajectory_minibatch['actions']]
            log_sampled_prev_action_probs = trajectory_minibatch['metadata']['log_action_probs'][
                minibatch_range, trajectory_minibatch['actions']]
            action_prob_ratio = jnp.exp(log_sampled_action_probs - log_sampled_prev_action_probs)
            actor_loss_1 = -action_prob_ratio * trajectory_minibatch['advantage']
            actor_loss_2 = -jnp.clip(
                action_prob_ratio,
                1.0 - self._config['ppo_eps'],
                1.0 + self._config['ppo_eps'],
            ) * trajectory_minibatch['advantage']
            ppo_loss = self._config['ppo_loss_weight'] * jnp.maximum(actor_loss_1, actor_loss_2).mean()

            value_function_loss = self._config['value_function_loss_weight'] * jnp.mean(
                0.5 * (state_values - trajectory_minibatch['value_targets']) ** 2)

            entropy_regularizer_loss = self._config['entropy_regularizer_weight'] * rlax.entropy_loss(
                log_action_probs, jnp.ones_like(trajectory_minibatch['advantage']))

            log_sampled_id_action_probs = log_id_action_probs[minibatch_range, trajectory_minibatch['actions']]
            inverse_dynamics_loss = -self._config['inverse_dynamics_loss_weight'] * jnp.mean(log_sampled_id_action_probs)

            return ppo_loss + value_function_loss + entropy_regularizer_loss + inverse_dynamics_loss, {
                'ppo_loss': ppo_loss,
                'value_function_loss': value_function_loss,
                'entropy_regularizer_loss': entropy_regularizer_loss,
                'inverse_dynamics_loss': inverse_dynamics_loss,
            }

        grad_and_stats_func = jax.grad(loss_function, argnums=0, has_aux=True)
        grads, stats = grad_and_stats_func(
            train_state.params, train_state, trajectory_minibatch, subkey1)
        if self._config['gradient_clipnorm'] is not None:
            grads = clip_gradient_by_norm(grads, self._config['gradient_clipnorm'])
        train_state = train_state.apply_gradients(grads=grads)

        if self._config['use_rnd']:
            def rnd_loss_function(rnd_params, rnd_train_state, trajectory_minibatch, rng):
                rnd_loss = rnd_train_state.apply_fn(
                    rnd_params, trajectory_minibatch['next_state'], deterministic=False, rng=rng)
                rnd_loss = jnp.mean(rnd_loss)
                return rnd_loss, {
                    'rnd_loss': rnd_loss,
                }

            rnd_grad_and_stats = jax.grad(rnd_loss_function, argnums=0, has_aux=True)
            rnd_grads, rnd_stats = rnd_grad_and_stats(
                rnd_train_state.params, rnd_train_state, trajectory_minibatch, subkey2)
            rnd_train_state = rnd_train_state.apply_gradients(grads=rnd_grads)
            stats = pytree.update(stats, rnd_stats)

        return train_state, rnd_train_state, stats

    def _sample_minibatch(self, trajectory_batch, rng):
        key1, key2 = jax.random.split(rng)

        minibatch_size = self._config['minibatch_size']
        num_trajectories = pytree.get_axis_dim(trajectory_batch, axis=0)
        num_timestamps = pytree.get_axis_dim(trajectory_batch, axis=1)

        trajectory_indices = jax.random.randint(key1, (minibatch_size,), 0, num_trajectories)
        timestamp_indices = jax.random.randint(key2, (minibatch_size,), 0, num_timestamps)

        # Note that we replace trajectory and timestamp dimensions by minibatch dimension here
        return jax.tree_map(
            lambda leaf: leaf[trajectory_indices, timestamp_indices, ...],
            trajectory_batch,
        )

    def _compute_advantage_and_value_targets(self, trajectory_batch):
        def per_trajectory_advantage(rewards, discounts, state_values):
            return rlax.truncated_generalized_advantage_estimation(
                rewards, discounts, self._config['gae_lambda'], state_values)
        per_batch_advantage = jax.vmap(per_trajectory_advantage, in_axes=0)

        discounts = ((1.0 - trajectory_batch['done']) * self._config['discount_factor'])[:, :-1]
        rewards = trajectory_batch['rewards'][:, :-1]
        if self._config['use_rnd']:
            # Taking RND loss aka surprise at the NEXT state as the exploration reward
            exploration_rewards = trajectory_batch['metadata']['rnd_loss'][:, 1:]
            # TODO: don't take episode end into accounts for extrinsic rewards (that would require a separate value head)
            rewards += self._config['exploration_reward_scale'] * exploration_rewards

        advantage = per_batch_advantage(
            rewards, discounts, trajectory_batch['metadata']['state_values'])
        value_targets = advantage + trajectory_batch['metadata']['state_values'][:, :-1]

        if self._config['normalize_advantage']:
            advantage = (advantage - jnp.mean(advantage)) / (jnp.std(advantage) + 1e-9)

        return advantage, value_targets

    def _preprocess_batch(self, trajectory_batch):
        advantage, value_targets = self._compute_advantage_and_value_targets(trajectory_batch)
        next_state = jax.tree_map(lambda l: l[:, 1:, ...], trajectory_batch['current_state'])

        # Get rid of states we don't have next states and GAE estimates for
        trajectory_batch = jax.tree_map(lambda l: l[:, :-1, ...], trajectory_batch)

        # Add the values we just computed to the batch
        trajectory_batch = pytree.update(trajectory_batch, {
            'advantage': advantage,
            'value_targets': value_targets,
            'next_state': next_state,
        })
        return trajectory_batch

    def _aggregate_train_stats(self, train_stats_per_minibatch, trajectory_batch):
        def aggregate_minibatch_stats(stats):
            return jnp.sum(stats) / self._config['num_minibatches_per_train_step']
        train_stats_minibatch_avg = jax.tree_map(f=aggregate_minibatch_stats, tree=train_stats_per_minibatch)
        batch_stats = {
            'state_value': jnp.mean(trajectory_batch['metadata']['state_values']),
            'advantage': jnp.mean(trajectory_batch['advantage']),
            'value_target': jnp.mean(trajectory_batch['value_targets']),
            'policy_entropy': jnp.mean(
                distributions.softmax().entropy(trajectory_batch['metadata']['log_action_probs']))
        }
        return pytree.update(batch_stats, train_stats_minibatch_avg)

    @partial(jax.jit, static_argnums=(0,))
    def _train_on_batch_jitted(self, train_state, rnd_train_state, trajectory_batch, rng):
        trajectory_batch = self._preprocess_batch(trajectory_batch)

        def train_step_on_minibatch_body(carry, nothing):
            rng, train_state, rnd_train_state = carry

            rng, subkey1, subkey2 = jax.random.split(rng, 3)
            trajectory_minibatch = self._sample_minibatch(trajectory_batch, subkey1)
            train_state, rnd_train_state, train_stats = self._train_on_minibatch(
                train_state, rnd_train_state, trajectory_minibatch, subkey2)

            return (rng, train_state, rnd_train_state), train_stats

        (rng, train_state, rnd_train_state), train_stats_per_minibatch = jax.lax.scan(
            f=train_step_on_minibatch_body,
            init=(rng, train_state, rnd_train_state),
            xs=None, length=self._config['num_minibatches_per_train_step'],
        )

        aggregated_train_stats = self._aggregate_train_stats(train_stats_per_minibatch, trajectory_batch)
        return train_state, rnd_train_state, aggregated_train_stats

    def act_on_batch(self, observation_batch):
        return self._act_on_batch_jitted(
            self._train_state, self._rnd_train_state, observation_batch, self.next_random_key())

    def train_on_batch(self, trajectory_batch):
        self._train_state, self._rnd_train_state, stats = self._train_on_batch_jitted(
            self._train_state, self._rnd_train_state, trajectory_batch, self.next_random_key())
        return stats
