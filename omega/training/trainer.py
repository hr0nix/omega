import abc
from functools import partial

import jax
import jax.numpy as jnp
import numpy as np

from ..training.batched_env_stepper import BatchedEnvStepper


class Trainer(abc.ABC):
    def __init__(self, env_factory, num_envs, num_collection_steps, num_workers):
        self._env_factory = env_factory
        self._num_collection_steps = num_collection_steps
        self._num_envs = num_envs

        self._batched_env_stepper = BatchedEnvStepper(self._env_factory, num_envs, num_workers)
        self._current_episode_indices = list(range(num_envs))
        self._next_episode_index = num_envs

    @property
    def num_collection_steps(self):
        return self._num_collection_steps

    def run_training_step(self, agent, stats=None):
        trajectory_batch = self._run_day(agent, stats)
        self._run_night(agent, stats, trajectory_batch)

    def _run_day(self, agent, stats):
        transition_batches = []

        for step in range(self.num_collection_steps):
            current_state_batch_cpu = self._batched_env_stepper.get_current_state()['current_state']
            action_batch_gpu, metadata_batch_gpu = agent.act_on_batch(current_state_batch_cpu)
            action_batch_cpu = np.asarray(action_batch_gpu)
            reward_done_next_state_batch_cpu = self._batched_env_stepper.step(action_batch_cpu)

            for env_index in range(self._num_envs):
                if stats is not None:
                    stats.add_transition(
                        self._current_episode_indices[env_index],
                        action_batch_cpu[env_index],
                        reward_done_next_state_batch_cpu['rewards'][env_index],
                        reward_done_next_state_batch_cpu['done'][env_index])

                if reward_done_next_state_batch_cpu['done'][env_index]:
                    # Allocate a new episode index for stats accumulation
                    self._current_episode_indices[env_index] = self._next_episode_index
                    self._next_episode_index += 1

            transition_batches.append(
                dict(
                    current_state=current_state_batch_cpu,
                    actions=action_batch_gpu,
                    metadata=metadata_batch_gpu,
                    **reward_done_next_state_batch_cpu,
                )
            )

        return self._batch_trajectories(transition_batches)

    @partial(jax.jit, static_argnums=(0,))
    def _batch_trajectories(self, transition_batches):
        return jax.tree_map(
            lambda *leaves: jnp.stack(leaves, axis=1),  # Stack along timestamp dimension
            *transition_batches,
        )

    @abc.abstractmethod
    def _run_night(self, agent, stats, collected_trajectories):
        pass


class OnPolicyTrainer(Trainer):
    def _run_night(self, agent, stats, collected_trajectories):
        training_stats = agent.train_on_batch(collected_trajectories)
        stats.add_rolling_stats(training_stats)
