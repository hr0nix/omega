import abc
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Union

import numpy as np

from ..utils.collections import LinearPrioritizedSampler, get_dict_slice
from ..utils import pytree
from ..utils.profiling import timeit


@dataclass(eq=True, frozen=True)
class ReplayBufferItem(object):
    id: Any = field(compare=True)
    trajectory: Any = field(compare=False)


class ReplayBufferStorage(object):
    def __init__(self, max_size: int):
        self._storage = deque(maxlen=max_size)
        self._id_to_item = {}

    def add(self, item: ReplayBufferItem) -> Union[ReplayBufferItem, None]:
        if item.id in self._id_to_item:
            raise ValueError('This item has already been added.')

        evicted_item = None
        if len(self._storage) == self._storage.maxlen:
            evicted_item = self.evict_left()

        self._storage.append(item)
        self._id_to_item[item.id] = item

        return evicted_item

    def find_by_id(self, id) -> Union[ReplayBufferItem, None]:
        return self._id_to_item.get(id)

    def evict_left(self) -> ReplayBufferItem:
        evicted_item = self._storage.popleft()
        del self._id_to_item[evicted_item.id]
        return evicted_item

    def __len__(self):
        return len(self._storage)

    def __getitem__(self, index):
        return self._storage[index]


# TODO: ideally you'd want to save replay buffer state so that loading from checkpoint restores state fully
class ReplayBuffer(abc.ABC):
    @abc.abstractmethod
    def add_trajectory(self, trajectory_id, trajectory, *, priority=None, **kwargs):
        pass

    @abc.abstractmethod
    def find_trajectory(self, trajectory_id):
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

    def update_priority(self, trajectory_id, priority):
        raise NotImplementedError('This replay buffer type does not support priorities')

    def get_stats(self):
        return {}


class FIFOReplayBuffer(ReplayBuffer):
    def __init__(self, buffer_size):
        self._buffer = ReplayBufferStorage(max_size=buffer_size)

    def add_trajectory(self, trajectory_id, trajectory, *, priority=None, **kwargs):
        if priority is not None:
            raise ValueError('Priorities are not supported with this replay buffer type')

        trajectory = pytree.to_cpu(trajectory)  # Avoid wasting GPU memory for replay buffer
        self._buffer.add(ReplayBufferItem(id=trajectory_id, trajectory=trajectory))

    def find_trajectory(self, trajectory_id):
        return self._buffer.find_by_id(trajectory_id)

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


class MaxAgeReplayBuffer(ReplayBuffer):
    @dataclass(eq=True, frozen=True)
    class _ReplayBufferItem(ReplayBufferItem):
        add_step: int = field(compare=False)

    def __init__(self, max_age: int, max_buffer_size: int):
        self._max_age = max_age
        self._buffer = ReplayBufferStorage(max_size=max_buffer_size)

    def _try_evict(self, current_step):
        if self.size > 0 and self._buffer[-1].add_step > current_step:
            raise ValueError('Current day must be monotonically increasing')

        while self.size > 0 and current_step - self._buffer[0].add_step > self._max_age:
            self._buffer.evict_left()

    def add_trajectory(self, trajectory_id, trajectory, *, priority=None, current_step=None, **kwargs):
        if priority is not None:
            raise ValueError('Priorities are not supported with this replay buffer type')
        if current_step is None:
            raise ValueError('Current step must be provided for this replay buffer type')

        self._try_evict(current_step)

        trajectory = pytree.to_cpu(trajectory)  # Avoid wasting GPU memory for replay buffer
        self._buffer.add(self._ReplayBufferItem(id=trajectory_id, trajectory=trajectory, add_step=current_step))

    def find_trajectory(self, trajectory_id):
        return self._buffer.find_by_id(trajectory_id)

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

        self._buffer = ReplayBufferStorage(max_size=buffer_size)
        self._sampler = LinearPrioritizedSampler(max_items=buffer_size, alpha=alpha, epsilon=epsilon)

        self._last_batch_good_trajectory_fraction = 0
        self._last_batch_avg_priority = 0.0
        self._num_good_trajectories_in_buffer = 0

    def _try_evict(self):
        if len(self._buffer) == self._max_size:
            evicted_item = self._buffer.evict_left()
            self._sampler.remove(evicted_item)
            if self._is_good_trajectory(evicted_item.trajectory):
                self._num_good_trajectories_in_buffer -= 1

    def add_trajectory(self, trajectory_id, trajectory, *, priority=None, **kwargs):
        self._validate_priority(priority)

        self._try_evict()
        trajectory = pytree.to_cpu(trajectory)  # Avoid wasting GPU memory for replay buffer
        item = ReplayBufferItem(id=trajectory_id, trajectory=trajectory)
        self._buffer.add(item)
        self._sampler.add(item, priority)
        self._num_good_trajectories_in_buffer += self._is_good_trajectory(trajectory)

    def find_trajectory(self, trajectory_id):
        return self._buffer.find_by_id(trajectory_id)

    def sample_trajectory_batch(self, batch_size):
        random_trajectories = self._sampler.sample(num_items=batch_size)
        self._update_stats(random_trajectories)
        return random_trajectories

    def update_priority(self, trajectory_id, priority):
        self._validate_priority(priority)

        item = self._buffer.find_by_id(trajectory_id)
        if item is None:
            raise ValueError('Cannot find a trajectory with the provided id.')
        self._sampler.update_priority(item, priority)

    @staticmethod
    def _validate_priority(priority):
        if priority is None:
            raise ValueError('Priority must be specified')
        if priority < 0:
            raise ValueError('Priority must be non-negative')

    def _is_good_trajectory(self, trajectory):
        return np.sum(trajectory['rewards']) >= self._good_total_reward_threshold

    def _update_stats(self, sampled_trajectories):
        self._last_batch_avg_priority = 0.0
        self._last_batch_good_trajectory_fraction = 0.0

        for item in sampled_trajectories:
            self._last_batch_good_trajectory_fraction += self._is_good_trajectory(item.trajectory)
            self._last_batch_avg_priority += self._sampler.get_priority(item)

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
        super(ClusteringReplayBuffer, self).__init__()
        self._buffers = [cluster_buffer_fn() for _ in range(num_clusters)]
        self._clustering_fn = clustering_fn

    def add_trajectory(self, trajectory_id, trajectory, **kwargs):
        cluster_id = self._get_trajectory_cluster(trajectory)
        self._buffers[cluster_id].add_trajectory(trajectory_id, trajectory, **kwargs)

    def find_trajectory(self, trajectory_id):
        for buffer in self._buffers:
            found_item = buffer.find_trajectory(trajectory_id)
            if found_item:
                return found_item
        return None

    def update_priority(self, trajectory_id, priority):
        for buffer in self._buffers:
            if buffer.find_trajectory(trajectory_id) is not None:
                buffer.update_priority(trajectory_id, priority)
                break

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

    if buffer_type == 'fifo':
        return FIFOReplayBuffer(buffer_size=config['buffer_size'])

    elif buffer_type == 'max_age':
        return MaxAgeReplayBuffer(max_age=config['max_age'], max_buffer_size=config['max_buffer_size'])

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
