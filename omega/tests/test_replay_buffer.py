import pytest

import jax.numpy as jnp
import numpy as np

from omega.training.replay_buffer import ReplayBufferStorage, ReplayBufferItem
from omega.utils import pytree


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