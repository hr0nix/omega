import numpy as np
import ray

from .env_wrapper import EnvWrapper
from ..utils import pytree


@ray.remote
class Worker(object):
    def __init__(self, env_factory, num_envs):
        self._envs = [EnvWrapper(env_factory()) for _ in range(num_envs)]
        for env in self._envs:
            env.reset()

    def get_current_state(self):
        current_states = [env.current_state for env in self._envs]
        return pytree.stack(current_states, axis=0)

    def step(self, action_batch):
        reward_per_env = []
        done_per_env = []
        next_state_per_env = []
        for env_index in range(len(self._envs)):
            next_state, reward, done, info = self._envs[env_index].step(action_batch[env_index])
            if done:
                self._envs[env_index].reset()
                # If it is the end of the episode, we return the first state of the next episode as the next state
                next_state = self._envs[env_index].current_state

            reward_per_env.append(reward)
            done_per_env.append(done)
            next_state_per_env.append(next_state)

        return {
            'rewards': np.asarray(reward_per_env, dtype=np.float64),
            'done': np.asarray(done_per_env, dtype=np.bool_),
            'next_state': pytree.stack(next_state_per_env, axis=0),
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

    def get_current_state(self):
        current_state_promises = [
            worker.get_current_state.remote() for worker in self._workers
        ]
        current_states = ray.get(current_state_promises)
        return pytree.concatenate(current_states, axis=0)

    def step(self, action_batch):
        prev_index = 0
        worker_result_promises = []
        for worker_index in range(len(self._workers)):
            worker_result_promise = self._workers[worker_index].step.remote(
                action_batch[prev_index:prev_index + self._num_envs_per_worker[worker_index]],
            )
            worker_result_promises.append(worker_result_promise)
            prev_index += self._num_envs_per_worker[worker_index]
        worker_results = ray.get(worker_result_promises)
        return pytree.concatenate(worker_results, axis=0)
