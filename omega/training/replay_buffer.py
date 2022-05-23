import abc
from collections import deque

import numpy as np

from ..utils.collections import LinearPrioritizedSampler, IdentityHashWrapper, get_dict_slice
from ..utils import pytree
from ..utils.profiling import timeit


# TODO: ideally you'd want to save replay buffer state so that loading from checkpoint restores state fully
class ReplayBuffer(abc.ABC):
    @abc.abstractmethod
    def add_trajectory(self, trajectory, priority=None):
        pass

    @abc.abstractmethod
    def sample_trajectory_batch(self, batch_size):
        pass

    @property
    @abc.abstractmethod
    def size(self):
        pass

    @property
    def empty(self):
        return self.size == 0

    def update_priority(self, trajectory, priority):
        raise NotImplementedError('This replay buffer type does not support priorities')

    def get_stats(self):
        return {}


class FIFOReplayBuffer(ReplayBuffer):
    def __init__(self, buffer_size):
        self._buffer = deque(maxlen=buffer_size)

    def add_trajectory(self, trajectory, priority=None):
        if priority is not None:
            raise ValueError('Priorities are not supported with this replay buffer type')

        trajectory = pytree.to_cpu(trajectory)  # Avoid wasting GPU memory for replay buffer
        self._buffer.append(trajectory)

    def sample_trajectory_batch(self, batch_size):
        random_indices = np.random.randint(low=0, high=len(self._buffer), size=batch_size)
        random_trajectories = [self._buffer[i] for i in random_indices]
        return random_trajectories

    @property
    def size(self):
        return len(self._buffer)

    def get_stats(self):
        return {
            'replay_buffer_size': len(self._buffer)
        }


class PrioritizedReplayBuffer(ReplayBuffer):
    def __init__(self, buffer_size, alpha=1.0, epsilon=1e-3, good_total_reward_threshold=0.5):
        self._max_size = buffer_size
        self._good_total_reward_threshold = good_total_reward_threshold

        self._buffer = deque(maxlen=buffer_size)
        self._sampler = LinearPrioritizedSampler(max_items=buffer_size, alpha=alpha, epsilon=epsilon)

        self._last_batch_good_trajectory_fraction = 0
        self._last_batch_avg_priority = 0.0
        self._num_good_trajectories_in_buffer = 0

    def add_trajectory(self, trajectory, priority=None):
        self._validate_priority(priority)

        trajectory = pytree.to_cpu(trajectory)  # Avoid wasting GPU memory for replay buffer
        if len(self._buffer) == self._max_size:
            evicted_trajectory = self._buffer.popleft()
            self._sampler.remove(self._for_sampler(evicted_trajectory))
            if self._is_good_trajectory(evicted_trajectory):
                self._num_good_trajectories_in_buffer -= 1
        self._buffer.append(trajectory)
        self._sampler.add(self._for_sampler(trajectory), priority)
        self._num_good_trajectories_in_buffer += self._is_good_trajectory(trajectory)

    def sample_trajectory_batch(self, batch_size):
        sampled_trajectories = [wrapper.val for wrapper in self._sampler.sample(num_items=batch_size)]
        self._update_stats(sampled_trajectories)
        return sampled_trajectories

    def update_priority(self, trajectory, priority):
        self._validate_priority(priority)
        self._sampler.update_priority(self._for_sampler(trajectory), priority)

    @staticmethod
    def _validate_priority(priority):
        if priority is None:
            raise ValueError('Priority must be specified')
        if priority < 0:
            raise ValueError('Priority must be non-negative')

    @staticmethod
    def _for_sampler(trajectory):
        # Sampler requires hashable item types, and trajectories are pytrees (dicts).
        # To work around this, we use a wrapper with value identity based hashing.
        return IdentityHashWrapper(trajectory)

    def _is_good_trajectory(self, trajectory):
        return np.sum(trajectory['rewards']) >= self._good_total_reward_threshold

    def _update_stats(self, sampled_trajectories):
        self._last_batch_avg_priority = 0.0
        self._last_batch_good_trajectory_fraction = 0.0

        for trajectory in sampled_trajectories:
            self._last_batch_good_trajectory_fraction += self._is_good_trajectory(trajectory)
            self._last_batch_avg_priority += self._sampler.get_priority(self._for_sampler(trajectory))

        batch_size = len(sampled_trajectories)
        self._last_batch_avg_priority /= batch_size
        self._last_batch_good_trajectory_fraction /= batch_size

    @property
    def size(self):
        return len(self._buffer)

    def get_stats(self):
        return {
            'replay_buffer_size': len(self._buffer),
            'last_batch_avg_priority': self._last_batch_avg_priority,
            'last_batch_good_trajectory_fraction': self._last_batch_good_trajectory_fraction,
            'num_good_trajectories_in_buffer': self._num_good_trajectories_in_buffer,
        }


class ClusteringReplayBuffer(ReplayBuffer):
    def __init__(self, cluster_buffer_fn, num_clusters, clustering_fn):
        self._buffers = [cluster_buffer_fn() for _ in range(num_clusters)]
        self._clustering_fn = clustering_fn

    def add_trajectory(self, trajectory, priority=None):
        cluster_id = self._get_trajectory_cluster(trajectory)
        self._buffers[cluster_id].add_trajectory(trajectory, priority)

    def update_priority(self, trajectory, priority):
        cluster_id = self._get_trajectory_cluster(trajectory)
        self._buffers[cluster_id].update_priority(trajectory, priority)

    def sample_trajectory_batch(self, batch_size):
        num_clusters = len(self._buffers)
        # Some clusters might still be empty because we didn't see trajectories that fit there
        active_clusters = set(
            cluster_id for cluster_id in range(num_clusters) if not self._buffers[cluster_id].empty
        )
        num_active_clusters = len(active_clusters)
        assert num_active_clusters > 0

        # Distribute batch size between all active clusters
        num_samples_from_cluster = [
            batch_size // num_active_clusters + (1 if cluster_id < batch_size % num_active_clusters else 0)
            for cluster_id in active_clusters
        ]
        assert sum(num_samples_from_cluster) == batch_size

        random_trajectories = [
            trajectory
            for active_cluster_idx, cluster_id in enumerate(active_clusters)
            for trajectory in self._buffers[cluster_id].sample_trajectory_batch(
                num_samples_from_cluster[active_cluster_idx])
        ]
        return random_trajectories

    def _get_trajectory_cluster(self, trajectory):
        cluster_id = self._clustering_fn(trajectory)
        if cluster_id < 0 or cluster_id >= len(self._buffers):
            raise RuntimeError(f'An unexpected replay buffer cluster index: {cluster_id}')
        return cluster_id

    @property
    def size(self):
        return sum(b.size for b in self._buffers)

    def get_stats(self):
        return {
            f'{key}_cluster_{cluster_id}': value
            for cluster_id in range(len(self._buffers))
            for key, value in self._buffers[cluster_id].get_stats().items()
        }


def create_from_config(config):
    buffer_type = config['type']

    if buffer_type == 'uniform':
        return FIFOReplayBuffer(buffer_size=config['buffer_size'])

    elif buffer_type == 'uniform_over_good_and_bad':
        clustering_fn = lambda t: 1 if np.sum(t['rewards']) >= config['good_total_reward_threshold'] else 0
        cluster_fn = lambda: create_from_config(config['cluster_buffer'])
        return ClusteringReplayBuffer(
            cluster_fn,
            num_clusters=2,
            clustering_fn=clustering_fn,
        )

    elif buffer_type == 'prioritized':
        return PrioritizedReplayBuffer(
            buffer_size=config['buffer_size'],
            **get_dict_slice(config, ['alpha', 'epsilon', 'good_total_reward_threshold'])
        )

    else:
        raise ValueError(f'Unknown replay buffer type: {buffer_type}')
