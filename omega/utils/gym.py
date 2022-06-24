import gym
import ray

import numpy as np

from . import pytree


class StayInTerminalStateWrapper(gym.Wrapper):
    """
    After transitioning to terminal state, allows to act there.
    Actions will result in zero reward and remaining in the terminal state.
    Done is postponed one step.
    """
    def __init__(self, env):
        super().__init__(env)
        self._is_in_terminal = None
        self._terminal_state = None

    def reset(self, **kwargs):
        self._is_in_terminal = False
        self._terminal_state = None
        return super().reset(**kwargs)

    def step(self, action):
        assert self._is_in_terminal is not None

        if not self._is_in_terminal:
            observation, reward, done, info = super().step(action)
            if done:
                self._is_in_terminal = True
                self._terminal_state = observation
                done = False
            return observation, reward, done, info
        else:
            assert self._terminal_state is not None
            return self._terminal_state, 0.0, True, {}


class AutoResetWrapper(gym.Wrapper):
    """
    Automatically resets the environment after transitioning to a terminal state.
    In that case the returned observation will be the initial state of the next episode.
    """
    def step(self, action):
        observation, reward, done, info = super().step(action)
        if done:
            observation = self.reset()
        return observation, reward, done, info


@ray.remote
class RayEnvWorker(object):
    def __init__(self, env_factory, num_envs):
        self._envs = [env_factory() for _ in range(num_envs)]

    def reset(self):
        start_state_per_env = []
        for env in self._envs:
            start_state = env.reset()
            start_state_per_env.append(start_state)
        return pytree.stack(start_state_per_env, axis=0)

    def step(self, action_batch):
        reward_per_env = []
        done_per_env = []
        next_state_per_env = []
        for env_index in range(len(self._envs)):
            next_state, reward, done, info = self._envs[env_index].step(action_batch[env_index])
            reward_per_env.append(reward)
            done_per_env.append(done)
            next_state_per_env.append(next_state)

        return {
            'rewards': np.asarray(reward_per_env, dtype=np.float64),
            'done': np.asarray(done_per_env, dtype=np.bool_),
            'next_state': pytree.stack(next_state_per_env, axis=0),
        }


class RayEnvStepper(object):
    def __init__(self, env_factory, num_envs, num_workers):
        self._num_envs_per_worker = [
            # Put all extra envs in the worker number zero
            num_envs // num_workers if worker_index > 0 else num_envs // num_workers + num_envs % num_workers
            for worker_index in range(num_workers)
        ]
        self._workers = [
            RayEnvWorker.remote(env_factory, num_envs=self._num_envs_per_worker[worker_index])
            for worker_index in range(num_workers)
        ]

    def reset(self):
        worker_result_promises = []
        for worker_index in range(len(self._workers)):
            worker_result_promise = self._workers[worker_index].reset.remote()
            worker_result_promises.append(worker_result_promise)
        worker_results = ray.get(worker_result_promises)
        return pytree.concatenate(worker_results, axis=0)

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