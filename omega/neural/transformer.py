from dataclasses import field
from typing import Optional, Dict

from flax import linen as nn

from .gating import GRUGate


class TransformerBlock(nn.Module):
    dim: int
    fc_inner_dim: int
    num_heads: int = 1
    dropout_rate: float = 0.1
    use_gating: bool = False
    gating_config: Dict = field(default_factory=dict)
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
        if self.use_gating:
            self._attention_gate = GRUGate(**self.gating_config)
            self._fc_gate = GRUGate(**self.gating_config)

    def __call__(self, queries, keys_values, deterministic=None):
        deterministic = nn.module.merge_param('deterministic', self.deterministic, deterministic)

        assert queries.shape[-1] == keys_values.shape[-1]
        l = queries

        l_prev = l
        q = self._att_norm_q(l)
        kv = self._att_norm_k(keys_values)
        l = self._att(q, kv)
        l = self._att_dropout(l, deterministic=deterministic)
        if self.use_gating:
            l = self._attention_gate(l_prev, l)
        else:
            l = l_prev + l

        l_prev = l
        l = self._fc_norm(l)
        l = self._fc_inner(l)
        l = nn.relu(l)
        l = self._fc(l)
        l = self._fc_dropout(l, deterministic=deterministic)
        if self.use_gating:
            l = self._fc_gate(l_prev, l)
        else:
            l = l_prev + l

        return l


class TransformerNetBase(nn.Module):
    num_blocks: int
    dim: int
    fc_inner_dim: int
    num_heads: int = 1
    dropout_rate: float = 0.1
    use_gating: bool = False
    gating_config: Dict = field(default_factory=dict)
    deterministic: Optional[bool] = None

    def setup(self):
        self._blocks = [
            TransformerBlock(
                dim=self.dim,
                fc_inner_dim=self.fc_inner_dim,
                num_heads=self.num_heads,
                dropout_rate=self.dropout_rate,
                use_gating=self.use_gating,
                gating_config=self.gating_config,
                deterministic=self.deterministic,
                name=f'block_{block_idx}',
            )
            for block_idx in range(self.num_blocks)
        ]


class TransformerNet(TransformerNetBase):
    def __call__(self, inputs, deterministic=None):
        deterministic = nn.module.merge_param('deterministic', self.deterministic, deterministic)

        l = inputs
        for block_idx in range(self.num_blocks):
            l = self._blocks[block_idx](queries=l, keys_values=l, deterministic=deterministic)
        return l


class CrossTransformerNet(TransformerNetBase):
    def __call__(self, queries, keys_values, deterministic=None):
        deterministic = nn.module.merge_param('deterministic', self.deterministic, deterministic)

        l = queries
        for block_idx in range(self.num_blocks):
            l = self._blocks[block_idx](queries=l, keys_values=keys_values, deterministic=deterministic)
        return l
