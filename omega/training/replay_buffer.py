import abc
from collections import deque
from functools import partial

import numpy as np
import jax

from ..utils import pytree
from ..utils.profiling import timeit


# TODO: ideally you'd want to save replay buffer state so that loading from checkpoint restores state fully
class ReplayBuffer(abc.ABC):
    @abc.abstractmethod
    def add_trajectory_batch(self, trajectory_batch):
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

    def get_stats(self):
        return {}


class FIFOReplayBuffer(ReplayBuffer):
    def __init__(self, buffer_size):
        self._buffer = deque(maxlen=buffer_size)

    def add_trajectory_batch(self, trajectory_batch):
        trajectory_batch = pytree.to_cpu(trajectory_batch)  # Avoid wasting GPU memory for replay buffer
        batch_size = pytree.get_axis_dim(trajectory_batch, axis=0)
        for trajectory_idx in range(batch_size):
            trajectory = pytree.slice_from_batch(trajectory_batch, trajectory_idx)
            assert pytree.is_cpu(trajectory)
            self._buffer.append(trajectory)

    def sample_trajectory_batch(self, batch_size):
        random_indices = np.random.randint(low=0, high=len(self._buffer), size=batch_size)
        random_trajectories = [self._buffer[i] for i in random_indices]
        return pytree.stack(random_trajectories, axis=0)

    @property
    def size(self):
        return len(self._buffer)

    def get_stats(self):
        return {
            'replay_buffer_size': len(self._buffer)
        }


class ClusteringReplayBuffer(ReplayBuffer):
    def __init__(self, cluster_buffer_size, num_clusters, clustering_fn):
        self._buffers = [FIFOReplayBuffer(cluster_buffer_size) for _ in range(num_clusters)]
        self._clustering_fn = clustering_fn

    def add_trajectory_batch(self, trajectory_batch):
        trajectory_batch = pytree.to_cpu(trajectory_batch)  # Avoid wasting GPU memory for replay buffer
        batch_size = pytree.get_axis_dim(trajectory_batch, axis=0)
        for trajectory_idx in range(batch_size):
            trajectory = pytree.slice_from_batch(trajectory_batch, trajectory_idx)
            cluster_id = self._clustering_fn(trajectory)
            assert 0 <= cluster_id < len(self._buffers)
            trajectory_as_batch = pytree.expand_dims(trajectory, axis=0)
            assert pytree.is_cpu(trajectory_as_batch)  # Make sure we didn't copy back to GPU by mistake
            self._buffers[cluster_id].add_trajectory_batch(trajectory_as_batch)

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

        batches_per_cluster = [
            self._buffers[cluster_id].sample_trajectory_batch(num_samples_from_cluster[active_cluster_idx])
            for active_cluster_idx, cluster_id in enumerate(active_clusters)
        ]
        return pytree.concatenate(batches_per_cluster, axis=0)

    @property
    def size(self):
        return sum(b.size for b in self._buffers)

    def get_stats(self):
        return {
            f'replay_buffer_size_cl_{cluster_id}': self._buffers[cluster_id].size
            for cluster_id in range(len(self._buffers))
        }
