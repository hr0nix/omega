import jax
import numpy as np
import ray

from .env_wrapper import EnvWrapper


@ray.remote
class Worker(object):
    def __init__(self, env_factory, num_envs):
        self._envs = [EnvWrapper(env_factory()) for _ in range(num_envs)]
        for env in self._envs:
            env.reset()

    @staticmethod
    def _stack_states(states):
        return jax.tree_map(
            lambda *leaves: np.stack(leaves, axis=0),  # Stack along batch axis
            *states,
        )

    def get_current_state(self):
        current_states = [env.current_state for env in self._envs]
        return {
            'current_state': self._stack_states(current_states),
        }

    def step(self, action_batch):
        reward_per_env = []
        done_per_env = []
        for env_index in range(len(self._envs)):
            obs_after, reward, done, info = self._envs[env_index].step(action_batch[env_index])
            reward_per_env.append(reward)
            done_per_env.append(done)
            if done:
                self._envs[env_index].reset()

        return {
            'reward': np.asarray(reward_per_env, dtype=np.float),
            'done': np.asarray(done_per_env, dtype=np.bool_),
        }


class BatchedEnvStepper(object):
    def __init__(self, env_factory, num_envs, num_workers):
        self._num_envs_per_worker = [
            # Put all extra envs in the worker number zero
            num_envs // num_workers if worker_index > 0 else num_envs // num_workers + num_envs % num_workers
            for worker_index in range(num_workers)
        ]
        self._workers = [
            Worker.remote(env_factory, num_envs=self._num_envs_per_worker[worker_index])
            for worker_index in range(num_workers)
        ]

    @staticmethod
    def _concat_worker_results(results):
        return jax.tree_map(
            lambda *leaves: np.concatenate(leaves, axis=0),  # Concatenate along batch axis
            *results,
        )

    def get_current_state(self):
        current_state_promises = [
            worker.get_current_state.remote() for worker in self._workers
        ]
        return self._concat_worker_results(ray.get(current_state_promises))

    def step(self, action_batch):
        prev_index = 0
        worker_result_promises = []
        for worker_index in range(len(self._workers)):
            worker_result_promise = self._workers[worker_index].step.remote(
                action_batch[prev_index:prev_index + self._num_envs_per_worker[worker_index]])
            worker_result_promises.append(worker_result_promise)
            prev_index += self._num_envs_per_worker[worker_index]
        return self._concat_worker_results(ray.get(worker_result_promises))
