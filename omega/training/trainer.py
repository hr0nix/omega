import abc
from functools import partial

import jax

from ..training.batched_env_stepper import BatchedEnvStepper
from ..utils import pytree
from ..utils.profiling import timeit


class Trainer(abc.ABC):
    def __init__(self, agent, env_factory, num_envs, num_collection_steps, num_workers):
        self._agent = agent
        self._env_factory = env_factory
        self._num_collection_steps = num_collection_steps
        self._num_envs = num_envs

        self._batched_env_stepper = BatchedEnvStepper(self._env_factory, num_envs, num_workers)
        self._current_episode_indices = list(range(num_envs))
        self._next_episode_index = num_envs

        self._agent_memory = self._agent.init_memory_batch(num_envs)

    @property
    def agent(self):
        return self._agent

    @property
    def num_collection_steps(self):
        return self._num_collection_steps

    def run_training_step(self, stats=None):
        # Collect some trajectories by interacting with the environment. We call it a "day" stage.
        trajectory_batch = self._run_day(stats)
        # Update the parameters of the agent. We call it a "night" stage.
        self._run_night(stats, trajectory_batch)

    def _run_day(self, stats):
        transition_batches = []

        for step in range(self.num_collection_steps):
            current_state_batch_cpu = self._batched_env_stepper.get_current_state()
            action_batch_gpu, metadata_batch_gpu = self.agent.act_on_batch(current_state_batch_cpu, self._agent_memory)
            # Copy actions back to CPU because indexing GPU memory will slow everything down significantly
            action_batch_cpu = pytree.to_cpu(action_batch_gpu)
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
                    memory_before=self._agent_memory,
                    current_state=current_state_batch_cpu,
                    actions=action_batch_gpu,
                    metadata=metadata_batch_gpu,
                    **reward_done_next_state_batch_cpu,
                )
            )

            self._agent_memory = self.agent.update_memory_batch(
                prev_memory=self._agent_memory,
                metadata=metadata_batch_gpu,
                actions=action_batch_gpu,
                done=reward_done_next_state_batch_cpu['done'],
                last_step_of_the_day=step == self.num_collection_steps - 1,
            )

        return self._stack_transition_batches(transition_batches)

    @partial(jax.jit, static_argnums=(0,))
    def _stack_transition_batches(self, transition_batches):
        # Stack along timestamp dimension, return as GPU tensors
        return pytree.stack(transition_batches, axis=1, result_device='gpu')

    @abc.abstractmethod
    def _run_night(self, stats, collected_trajectories):
        pass


class DummyTrainer(Trainer):
    def _run_night(self, agent, stats, collected_trajectories):
        pass


class OnPolicyTrainer(Trainer):
    def _run_night(self, stats, collected_trajectories):
        training_stats = self.agent.train_on_batch(collected_trajectories)
        stats.add_rolling_stats(training_stats)
