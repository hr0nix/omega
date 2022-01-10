from typing import Optional

import jax.random
from flax import linen as nn


class TransformerBlock(nn.Module):
    dim: int
    fc_inner_dim: int
    num_heads: int = 1
    dropout_rate: float = 0.1
    deterministic: Optional[bool] = None

    def setup(self):
        assert self.dim % self.num_heads == 0
        self._att = nn.MultiHeadDotProductAttention(
            num_heads=self.num_heads,
            qkv_features=self.dim,
            out_features=self.dim)
        self._fc_inner = nn.Dense(self.fc_inner_dim)
        self._fc = nn.Dense(self.dim)
        self._att_norm_q = nn.LayerNorm()
        self._att_norm_k = nn.LayerNorm()
        self._att_norm_v = nn.LayerNorm()
        self._fc_norm = nn.LayerNorm()
        self._att_dropout = nn.Dropout(rate=self.dropout_rate, deterministic=self.deterministic)
        self._fc_dropout = nn.Dropout(rate=self.dropout_rate, deterministic=self.deterministic)

    def __call__(self, queries, keys_values, deterministic=None, rng=None):
        deterministic = nn.module.merge_param('deterministic', self.deterministic, deterministic)

        if rng is None:
            rng = self.make_rng('transformer_block')

        assert queries.shape[-1] == keys_values.shape[-1]
        l = queries

        l_prev = l
        q = self._att_norm_q(l)
        kv = self._att_norm_k(keys_values)
        l = self._att(q, kv)
        rng, subkey = jax.random.split(rng)
        l = self._att_dropout(l, deterministic=deterministic, rng=subkey)
        l = l_prev + l

        l_prev = l
        l = self._fc_norm(l)
        l = self._fc_inner(l)
        l = nn.relu(l)
        l = self._fc(l)
        rng, subkey = jax.random.split(rng)
        l = self._fc_dropout(l, deterministic=deterministic, rng=subkey)
        l = l_prev + l

        return l


class TransformerNetBase(nn.Module):
    num_blocks: int
    dim: int
    fc_inner_dim: int
    num_heads: int = 1
    dropout_rate: float = 0.1
    deterministic: Optional[bool] = None

    def setup(self):
        self._blocks = [
            TransformerBlock(
                dim=self.dim,
                fc_inner_dim=self.fc_inner_dim,
                num_heads=self.num_heads,
                dropout_rate=self.dropout_rate,
                name=f'block_{block_idx}',
                deterministic=self.deterministic
            )
            for block_idx in range(self.num_blocks)
        ]


class TransformerNet(TransformerNetBase):
    def __call__(self, inputs, deterministic=None, rng=None):
        deterministic = nn.module.merge_param('deterministic', self.deterministic, deterministic)

        l = inputs
        for block_idx in range(self.num_blocks):
            l = self._blocks[block_idx](
                queries=l, keys_values=l, deterministic=deterministic, rng=rng
            )
        return l


class CrossTransformerNet(TransformerNetBase):
    def __call__(self, queries, keys_values, deterministic=None, rng=None):
        deterministic = nn.module.merge_param('deterministic', self.deterministic, deterministic)

        l = queries
        for block_idx in range(self.num_blocks):
            l = self._blocks[block_idx](
                queries=l, keys_values=keys_values, deterministic=deterministic, rng=rng
            )
        return l
