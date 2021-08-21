import flax.linen as nn


class DenseNet(nn.Module):
    num_blocks: int
    dim: int
    output_dim: int
    activation: str = 'relu'

    def setup(self):
        self._input_dense = nn.Dense(
            self.dim,
            name='{}/input_dense'.format(self.name)
        )
        self._dense = [
            nn.Dense(
                self.dim,
                name='{}/dense_{}'.format(self.name, block_idx)
            )
            for block_idx in range(self.num_blocks)
        ]
        self._output_dense = nn.Dense(
            self.output_dim,
            name='{}/output_dense'.format(self.name)
        )
        self._norm = [
            nn.LayerNorm(
                name='{}/norm_{}'.format(self.name, block_idx)
            )
            for block_idx in range(self.num_blocks)
        ]
        self._final_norm = nn.LayerNorm(
            name='{}/final_norm'.format(self.name)
        )

    def _activation(self, input):
        if self.activation == 'relu':
            return nn.relu(input)
        else:
            raise ValueError('Unknown activation: {}'.format(self.activation))

    def __call__(self, input):
        t = self._input_dense(input)
        for block_idx in range(self.num_blocks):
            t_prev = t
            t = self._dense[block_idx](t)
            t = self._activation(t)
            t = self._norm[block_idx](t)
            t += t_prev
        t = self._final_norm(t)
        return self._output_dense(t)
