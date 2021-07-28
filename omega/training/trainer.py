import abc

import jax.numpy as jnp

from omega.training.env_wrapper import EnvWrapper

from .trajectory import TrajectoryBatch
from ..evaluation import EvaluationStats


class Trainer(abc.ABC):
    def __init__(self, env_factory, num_parallel_envs, num_day_steps, num_night_steps):
        self._env_factory = env_factory
        self._num_day_steps = num_day_steps
        self._num_night_steps = num_night_steps

        self._training_envs = [
            EnvWrapper(env_factory())
            for _ in range(num_parallel_envs)
        ]
        for env in self._training_envs:
            env.reset()

    @property
    def num_day_steps(self):
        return self._num_day_steps

    @property
    def num_night_steps(self):
        return self._num_night_steps

    def run_training_step(self, agent, stats=None):
        trajectory_batch, stats = self._run_day(agent, stats)
        self._run_night(agent, trajectory_batch)
        return stats

    def _run_day(self, agent, stats=None):
        trajectory_batch = TrajectoryBatch()
        trajectories = [
            trajectory_batch.add_trajectory(env.current_state)
            for env in self._training_envs
        ]
        run_indices = [i for i in range(len(self._training_envs))]
        next_run_index = len(self._training_envs)
        stats = stats or EvaluationStats()

        for step in range(self.num_day_steps):
            observation_batch = self._batch_tensors([env.current_state for env in self._training_envs])
            action_batch = agent.act(observation_batch)

            for env_index in range(len(self._training_envs)):
                obs, reward, done, info = self._training_envs[env_index].step(action_batch[env_index])
                trajectories[env_index].append(action_batch[env_index], obs, reward, done)
                stats.add_stats(run_indices[env_index], reward)
                if done:
                    # Start a new trajectory
                    trajectories[env_index] = trajectory_batch.add_trajectory(
                        self._training_envs[env_index].reset())
                    # Allocate a new run index for stats accumulation
                    run_indices[env_index] = next_run_index
                    next_run_index += 1

        return trajectory_batch, stats

    @abc.abstractmethod
    def _run_night(self, agent, daytime_trajectories):
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

    def _run_night(self, agent, daytime_trajectories):
        for _ in range(self.num_night_steps):
            batch = daytime_trajectories.sample_subbatch(self._batch_size)
            agent.train_on_batch(batch)
