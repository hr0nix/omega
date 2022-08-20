import flax.linen as nn

from typing import Dict, Any
from dataclasses import field

Array = Any


class DropoutEnsemble(nn.Module):
    element_type: nn.Module = None
    element_config: Dict = field(default_factory=lambda: {})

    @nn.compact
    def __call__(self, inputs, num_elements):
        element_ensemble = nn.vmap(
            self.element_type,
            variable_axes={'params': None},
            in_axes=None, out_axes=0, axis_size=num_elements,
            split_rngs={'params': False, 'dropout': True}
        )
        return element_ensemble(**self.element_config)(inputs)

