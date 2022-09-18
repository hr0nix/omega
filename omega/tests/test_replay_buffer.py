import pytest  # noqa

import jax.numpy as jnp
import numpy as np

from omega.training.replay_buffer import ReplayBufferStorage, ReplayBufferItem, ClusteringReplayBuffer, FIFOReplayBuffer
from omega.utils import pytree


def test_clustering_replay_sampling():
    buffer = ClusteringReplayBuffer(
        cluster_buffer_fn=lambda: FIFOReplayBuffer(buffer_size=3),
        num_clusters=2,
        clustering_fn=lambda item: 1 if item > 0 else 0
    )
    # 1 trajectory in the first cluster, 3 in the second
    for trajectory_id in range(4):
        buffer.add_trajectory(trajectory_id, trajectory_id)

    trajectories, weights = buffer.sample_trajectory_batch(4)
    for i in range(4):
        if trajectories[i].trajectory == 0:
            assert weights[i] == 0.5  # 2 trajectories in cluster 0 contribute 1/4
        else:
            assert weights[i] == 1.5  # 2 trajectories in cluster 1 contribute 3/4


def test_replay_buffer_storage_serialization(tmp_path):
    buffer_path = tmp_path / 'buffer'

    storage = ReplayBufferStorage(max_size=10)
    trajectory = {
        'np': np.asarray(1),
        'jnp': jnp.asarray(2),
    }
    item = ReplayBufferItem(id=1, trajectory=trajectory)
    storage.add(item)
    storage.save(buffer_path)

    deserialized_storage = ReplayBufferStorage.load(buffer_path)
    assert len(deserialized_storage) == 1
    assert pytree.get_schema(deserialized_storage[0].trajectory) == pytree.get_schema(trajectory)


def test_empty_replay_buffer_storage_serialization(tmp_path):
    buffer_path = tmp_path / 'buffer'

    storage = ReplayBufferStorage(max_size=10)
    storage.save(buffer_path)

    deserialized_storage = ReplayBufferStorage.load(buffer_path)
    assert len(deserialized_storage) == 0
