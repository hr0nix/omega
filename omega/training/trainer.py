import abc

import jax.numpy as jnp

from ..training.env_wrapper import EnvWrapper
from ..utils.profiling import timeit
from .trajectory import TrajectoryBatch
from ..evaluation import EvaluationStats


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
        trajectory_batch, stats = self._run_day(agent, stats)
        self._run_night(agent, trajectory_batch)
        return stats

    def _run_day(self, agent, stats=None):
        stats = stats or EvaluationStats()
        trajectory_batch = TrajectoryBatch(
            num_trajectories=len(self._training_envs), num_transitions=self.num_collection_steps)

        for step in range(self.num_collection_steps):
            observation_batch = self._batch_tensors([env.current_state for env in self._training_envs])
            action_batch, metadata_batch = agent.act(observation_batch)
            rewards, dones = [], []

            for env_index in range(len(self._training_envs)):
                obs_after, reward, done, info = self._training_envs[env_index].step(action_batch[env_index])
                rewards.append(reward)
                dones.append(done)

                stats.add_stats(self._env_run_indices[env_index], reward)

                if done:
                    # Start a new trajectory
                    self._training_envs[env_index].reset()

                    # Allocate a new run index for stats accumulation
                    self._env_run_indices[env_index] = self._next_env_run_index + 1
                    self._next_env_run_index += 1

            trajectory_batch.add_transition_batch(
                transition_index=step,
                observation=observation_batch,
                action=action_batch,
                reward=rewards,
                done=dones,
                metadata=metadata_batch,
            )

        return trajectory_batch, stats

    @abc.abstractmethod
    def _run_night(self, agent, collected_trajectories):
        pass

    @staticmethod
    def _batch_tensors(tensor_dicts):
        assert len(tensor_dicts) > 0
        tensor_keys = tensor_dicts[0].keys()
        assert all(td.keys() == tensor_keys for td in tensor_dicts)
        assert all(
            td[key].shape == tensor_dicts[0][key].shape
            for key in tensor_keys
            for td in tensor_dicts
        )

        return {
            key: jnp.stack([td[key] for td in tensor_dicts], axis=0)
            for key in tensor_keys
        }


class OnPolicyTrainer(Trainer):
    def __init__(self, batch_size, **kwargs):
        super(OnPolicyTrainer, self).__init__(**kwargs)
        self._batch_size = batch_size

    def _run_night(self, agent, collected_trajectories):
        agent.train_on_batch(collected_trajectories)
