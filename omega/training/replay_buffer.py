import abc
from collections import deque

import jax
import numpy as np


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
    def empty(self):
        pass


# TODO: come up with something more efficient
class FIFOReplayBuffer(ReplayBuffer):
    def __init__(self, buffer_size):
        self._buffer = deque(maxlen=buffer_size)

    def add_trajectory_batch(self, trajectory_batch):
        batch_size = jax.tree_leaves(trajectory_batch)[0].shape[0]

        # Copy trajectory batch to CPU to avoid wasting GPU memory for replay buffer
        trajectory_batch  = jax.tree_map(lambda t: np.asarray(t), trajectory_batch)

        for trajectory_idx in range(batch_size):
            trajectory = jax.tree_map(lambda t: t[trajectory_idx, ...], trajectory_batch)
            self._buffer.append(trajectory)

    def sample_trajectory_batch(self, batch_size):
        random_indices = np.random.randint(low=0, high=len(self._buffer), size=batch_size)
        flat_batch = [self._buffer[i] for i in random_indices]
        return jax.tree_map(
            lambda *leaves: np.stack(leaves, axis=0),  # Stack along batch axis
            *flat_batch,
        )

    @property
    def empty(self):
        return len(self._buffer) == 0
