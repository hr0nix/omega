import jax.numpy as jnp
import flax
import jax.ops


@jax.jit
def _update_buffers(
        transition_index, observation, action, reward, done, metadata,
        action_buffer, rewards_buffer, done_buffer, observation_buffers, metadata_buffers):

    def _add_transition_batch(buffer, idx, value):
        value = jnp.asarray(value, dtype=buffer.dtype)
        assert len(value.shape) == len(buffer.shape) - 1
        assert value.shape[0] == buffer.shape[0]
        assert value.shape[1:] == buffer.shape[2:], 'Provided value has an unexpected shape'

        return jax.ops.index_update(x=buffer, idx=jax.ops.index[:, idx, ...], y=value)

    def _add_transition_batch_from_tensor_bag(buffers, idx, tensor_bag):
        assert set(buffers.keys()) == set(tensor_bag.keys()), 'An unexpected set of keys has been provided'

        return {
            key: _add_transition_batch(buffer, idx, tensor_bag[key])
            for key, buffer in buffers.items()
        }

    return (
        _add_transition_batch(action_buffer, transition_index, action),
        _add_transition_batch(rewards_buffer, transition_index, reward),
        _add_transition_batch(done_buffer, transition_index, done),
        _add_transition_batch_from_tensor_bag(observation_buffers, transition_index, observation),
        _add_transition_batch_from_tensor_bag(metadata_buffers, transition_index, metadata),
    )


class TrajectoryBatch(object):
    def __init__(self, num_trajectories, num_transitions):
        self._num_trajectories = num_trajectories
        self._num_transitions = num_transitions

        self._actions = self._create_buffer(num_trajectories, num_transitions, element_shape=(), dtype=jnp.int32)
        self._rewards = self._create_buffer(num_trajectories, num_transitions, element_shape=(), dtype=jnp.float32)
        self._done = self._create_buffer(num_trajectories, num_transitions, element_shape=(), dtype=jnp.bool_)

        self._observation_buffers = None
        self._metadata_buffers = None

    @property
    def num_trajectories(self):
        return self._num_trajectories

    @property
    def num_transitions(self):
        return self._num_transitions

    def _create_buffer(self, num_trajectories, num_transitions, element_shape, dtype):
        return jnp.zeros(shape=(num_trajectories, num_transitions) + element_shape, dtype=dtype)

    def _create_buffers_for_tensor_bag(self, tensor_bag):
        return {
            key: self._create_buffer(self._num_trajectories, self._num_transitions, value.shape[1:], value.dtype)
            for key, value in tensor_bag.items()
        }

    def add_transition_batch(self, transition_index, observation, action, reward, done, metadata):
        assert 0 <= transition_index < self.num_transitions, 'Provided transition index is out of bounds'

        if self._observation_buffers is None:
            self._observation_buffers = self._create_buffers_for_tensor_bag(observation)
        if self._metadata_buffers is None:
            self._metadata_buffers = self._create_buffers_for_tensor_bag(metadata)

        (
            self._actions, self._rewards, self._done,
            self._observation_buffers, self._metadata_buffers,
        ) = _update_buffers(
            transition_index, observation, action, reward, done, metadata,
            self._actions, self._rewards, self._done,
            self._observation_buffers, self._metadata_buffers,
        )

    def to_dict(self):
        return flax.core.frozen_dict.FrozenDict({
            'actions': self._actions,
            'rewards': self._rewards,
            'done': self._done,
            'observations': self._observation_buffers,
            'metadata': self._metadata_buffers,
        })

