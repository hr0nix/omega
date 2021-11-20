import abc
from functools import partial

import jax
import numpy as np

from ..training.env_wrapper import EnvWrapper
from ..utils.pytree import stack
from .trajectory import TrajectoryBatch


class Trainer(abc.ABC):
    def __init__(self, env_factory, num_parallel_envs, num_collection_steps):
        self._env_factory = env_factory
        self._num_collection_steps = num_collection_steps

        self._init_training_envs(num_parallel_envs)

    def _init_training_envs(self, num_parallel_envs):
        self._training_envs = [
            EnvWrapper(self._env_factory())
            for _ in range(num_parallel_envs)
        ]
        for env in self._training_envs:
            env.reset()
        self._env_run_indices = list(range(len(self._training_envs)))
        self._next_env_run_index = len(self._training_envs)

    @property
    def num_collection_steps(self):
        return self._num_collection_steps

    def run_training_step(self, agent, stats=None):
        trajectory_batch = self._run_day(agent, stats)
        self._run_night(agent, stats, trajectory_batch)

    def _run_day(self, agent, stats):
        num_training_envs = len(self._training_envs)

        trajectory_batch = TrajectoryBatch(
            num_trajectories=num_training_envs, num_transitions=self.num_collection_steps)

        reward_batch = np.zeros(shape=(num_training_envs,), dtype=np.float)
        done_batch = np.zeros(shape=(num_training_envs,), dtype=np.bool_)

        for step in range(self.num_collection_steps):
            observation_batch = self._batch_tensors([env.current_state for env in self._training_envs])
            action_batch, metadata_batch = agent.act_on_batch(observation_batch)


            for env_index in range(num_training_envs):
                obs_after, reward, done, info = self._training_envs[env_index].step(action_batch[env_index])
                reward_batch[env_index] = reward
                done_batch[env_index] = done

                if stats is not None:
                    stats.add_transition(self._env_run_indices[env_index], action_batch[env_index], reward, done)

                if done:
                    # Start a new trajectory
                    self._training_envs[env_index].reset()

                    # Allocate a new run index for stats accumulation
                    self._env_run_indices[env_index] = self._next_env_run_index + 1
                    self._next_env_run_index += 1

            trajectory_batch.add_transition_batch(
                transition_index=step,
                pytree=dict(
                    observations=observation_batch,
                    actions=action_batch,
                    rewards=reward_batch,
                    done=done_batch,
                    metadata=metadata_batch,
                )
            )

        return trajectory_batch

    @abc.abstractmethod
    def _run_night(self, agent, stats, collected_trajectories):
        pass

    @partial(jax.jit, static_argnums=(0,))
    def _batch_tensors(self, tensor_dicts):
        return stack(tensor_dicts)


class OnPolicyTrainer(Trainer):
    def _run_night(self, agent, stats, collected_trajectories):
        training_stats = agent.train_on_batch(collected_trajectories)
        stats.add_rolling_stats(training_stats)
