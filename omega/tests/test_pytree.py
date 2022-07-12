import jaxlib.xla_extension
import numpy as np
import jax.numpy as jnp

from omega.utils import pytree


def test_get_schema():
    schema = pytree.get_schema({'a': np.asarray(1), 'b': jnp.asarray(2)})
    assert schema == {'a': np.ndarray, 'b': jaxlib.xla_extension.DeviceArray}


def test_restore_schema():
    schema = {'a': np.ndarray, 'b': jaxlib.xla_extension.DeviceArray}
    tree = {'a': np.asarray(1), 'b': np.asarray(2)}
    restored_tree = pytree.restore_schema(tree, schema)
    assert pytree.get_schema(restored_tree) == schema
