from functools import partial

from absl import logging

import flax.training.train_state
import flax.training.checkpoints
import jax.numpy as jnp
import jax.random
import optax
import rlax
from rlax._src import distributions

from ..utils.pytree import dict_update
from .trainable_agent import TrainableAgentBase
from ..neural.optimization import clip_gradient_by_norm


class NethackPPOAgent(TrainableAgentBase):
    CONFIG = flax.core.frozen_dict.FrozenDict({
        'value_function_loss_weight': 1.0,
        'entropy_regularizer_weight': 0.0,
        'lr': 1e-3,
        'discount_factor': 0.99,
        'gae_lambda': 0.95,
        'gradient_clipnorm': None,
        'num_minibatches_per_train_step': 100,
        'minibatch_size': 64,
        'ppo_eps': 0.25,
        'model_config': {}
    })

    class TrainState(flax.training.train_state.TrainState):
        step_index: int = None

    def __init__(self, model_factory, *args, config=None, **kwargs):
        super(NethackPPOAgent, self).__init__(*args, **kwargs)

        self._random_key = jax.random.PRNGKey(31337)
        self._config = self.CONFIG.copy(config or {})
        self._model_factory = model_factory
        self._model, self._train_state = self._build_model()

    def _next_random_key(self):
        self._random_key, subkey = jax.random.split(self._random_key)
        return subkey

    def _build_model(self):
        observations_batch = {
            key: jnp.zeros(
                shape=(1,) + desc.shape,
                dtype=desc.dtype
            )
            for key, desc in self.observation_space.spaces.items()
        }
        return self._build_model_for_batch(observations_batch)

    def _build_model_for_batch(self, observations_batch):
        model = self._model_factory(num_actions=self.action_space.n, **self._config['model_config'])
        model_params = model.init(
            self._next_random_key(), observations_batch, deterministic=False, rng=self._next_random_key())

        optimizer = optax.rmsprop(learning_rate=self._config['lr'])

        train_state = self.TrainState.create(apply_fn=model.apply, params=model_params, tx=optimizer)

        return model, train_state

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
            return self._train_state.step_index + 1

    def save_to_checkpoint(self, path, step):
        self._train_state = self._train_state.replace(step_index=step)
        flax.training.checkpoints.save_checkpoint(
            path, self._train_state, step=step, keep=1, overwrite=True)

    @partial(jax.jit, static_argnums=(0,))
    def _forward_step(self, train_state, observation_batch, rng):
        rng1, rng2 = jax.random.split(rng)
        log_action_probs, state_values = train_state.apply_fn(
            train_state.params, observation_batch, deterministic=True, rng=rng1)
        metadata = {
            'log_action_probs': log_action_probs,
            'state_values': state_values,
        }
        selected_actions = jax.random.categorical(rng2, log_action_probs)
        return selected_actions, metadata

    @partial(jax.jit, static_argnums=(0,))
    def _train_on_minibatch(self, train_state, trajectory_minibatch, rng):
        def loss_function(params, train_state, trajectory_minibatch, rng):
            log_action_probs, state_values = train_state.apply_fn(
                params, trajectory_minibatch['observations'], deterministic=False, rng=rng)

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
            ppo_loss = jnp.maximum(actor_loss_1, actor_loss_2).mean()

            value_function_loss = self._config['value_function_loss_weight'] * jnp.mean(
                0.5 * (state_values - trajectory_minibatch['value_targets']) ** 2)

            entropy_regularizer_loss = self._config['entropy_regularizer_weight'] * rlax.entropy_loss(
                log_action_probs, jnp.ones_like(trajectory_minibatch['advantage']))

            return ppo_loss + value_function_loss + entropy_regularizer_loss, {
                'ppo_loss': ppo_loss,
                'value_function_loss': value_function_loss,
                'entropy_regularizer_loss': entropy_regularizer_loss,
            }

        grad_and_stats = jax.grad(loss_function, argnums=0, has_aux=True)
        grads, stats = grad_and_stats(train_state.params, train_state, trajectory_minibatch, rng)

        if self._config['gradient_clipnorm'] is not None:
            grads = clip_gradient_by_norm(grads, self._config['gradient_clipnorm'])

        return train_state.apply_gradients(grads=grads), stats

    @partial(jax.jit, static_argnums=(0,))
    def _sample_minibatch(self, trajectory_batch, rng):
        key1, key2 = jax.random.split(rng)

        minibatch_size = self._config['minibatch_size']
        some_tensor = jax.tree_leaves(trajectory_batch)[0]
        num_trajectories = some_tensor.shape[0]
        num_timestamps = some_tensor.shape[1]

        trajectory_indices = jax.random.randint(key1, (minibatch_size,), 0, num_trajectories)
        timestamp_indices = jax.random.randint(key2, (minibatch_size,), 0, num_timestamps)

        # Note that we replace trajectory and timestamp dimensions by minibatch dimension here
        return jax.tree_map(
            lambda leaf: leaf[trajectory_indices, timestamp_indices, ...],
            trajectory_batch,
        )

    @partial(jax.jit, static_argnums=(0,))
    def _compute_advantage_and_value_targets(self, trajectory_batch):
        def per_trajectory_advantage(rewards, discounts, state_values):
            return rlax.truncated_generalized_advantage_estimation(
                rewards[:-1], discounts[:-1], self._config['gae_lambda'], state_values)

        per_batch_advantage = jax.vmap(per_trajectory_advantage, in_axes=0)
        discounts = (1.0 - trajectory_batch['done']) * self._config['discount_factor']
        advantage = per_batch_advantage(
            trajectory_batch['rewards'], discounts, trajectory_batch['metadata']['state_values'])
        value_targets = advantage + trajectory_batch['metadata']['state_values'][:, :-1]

        if self._config['normalize_advantage']:
            advantage = (advantage - jnp.mean(advantage)) / (jnp.std(advantage) + 1e-9)

        return advantage, value_targets

    def _preprocess_batch(self, trajectory_batch):
        advantage, value_targets = self._compute_advantage_and_value_targets(trajectory_batch)
        # Get rid of states we don't have GAE estimates for
        trajectory_batch = jax.tree_map(lambda l: l[:, :-1, ...], trajectory_batch)
        trajectory_batch = dict_update(trajectory_batch, {
            'advantage': advantage,
            'value_targets': value_targets,
        })
        return trajectory_batch

    def _train_step(self, train_state, trajectory_batch, rng):
        trajectory_batch = self._preprocess_batch(trajectory_batch)
        train_stats_sum = None
        for _ in range(self._config['num_minibatches_per_train_step']):
            rng, subkey1, subkey2 = jax.random.split(rng, 3)
            trajectory_minibatch = self._sample_minibatch(trajectory_batch, subkey1)
            train_state, train_stats = self._train_on_minibatch(train_state, trajectory_minibatch, subkey2)
            train_stats_sum = train_stats if train_stats_sum is None else jax.tree_map(
                jnp.add, train_stats_sum, train_stats)

        # TODO: jit stats computation?
        stats = {
            'state_value': jnp.mean(trajectory_batch['metadata']['state_values']),
            'advantage': jnp.mean(trajectory_batch['advantage']),
            'value_target': jnp.mean(trajectory_batch['value_targets']),
            'policy_entropy': jnp.mean(
                distributions.softmax().entropy(trajectory_batch['metadata']['log_action_probs']))
        }
        stats = dict_update(stats, {
            k: v / self._config['num_minibatches_per_train_step']
            for k, v in train_stats_sum.items()
        })

        return train_state, stats

    def act(self, observation_batch):
        return self._forward_step(self._train_state, observation_batch, self._next_random_key())

    def train_on_batch(self, trajectory_batch):
        self._train_state, stats = self._train_step(self._train_state, trajectory_batch.buffer, self._next_random_key())
        return stats
