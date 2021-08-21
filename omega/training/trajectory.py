import jax.numpy as jnp
import jax.ops
import jax.tree_util


@jax.jit
def _update_buffer(buffer, transition_index, pytree):
    def _update_single_buffer(buffer_leaf, pytree_leaf):
        pytree_leaf = jnp.asarray(pytree_leaf)
        return jax.ops.index_update(x=buffer_leaf, idx=jax.ops.index[:, transition_index, ...], y=pytree_leaf)
    return jax.tree_util.tree_map(_update_single_buffer, buffer, pytree)


class TrajectoryBatch(object):
    def __init__(self, num_trajectories, num_transitions):
        self._num_trajectories = num_trajectories
        self._num_transitions = num_transitions
        self._buffer = None

    @property
    def num_trajectories(self):
        return self._num_trajectories

    @property
    def num_transitions(self):
        return self._num_transitions

    @property
    def buffer(self):
        return self._buffer

    def add_transition_batch(self, transition_index, pytree):
        assert 0 <= transition_index < self.num_transitions, 'Provided transition index is out of bounds'

        if self._buffer is None:
            def _create_single_buffer(tensor):
                tensor = jnp.asarray(tensor)
                return jnp.zeros(
                    shape=(self.num_trajectories, self.num_transitions) + tensor.shape[1:],
                    dtype=tensor.dtype
                )
            self._buffer = jax.tree_util.tree_map(_create_single_buffer, pytree)

        self._buffer = _update_buffer(self._buffer, transition_index, pytree)
