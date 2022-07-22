from typing import Optional, Tuple

import flax.linen as nn
import jax.numpy as jnp

import nle.nethack

from omega.neural import TransformerNet, CrossTransformerNet, DenseNet
from omega.math import log_transform

from .base import ItemEmbedder


class PerceiverNethackStateEncoder(nn.Module):
    """
    Encodes a nethack state observation into a latent memory vector.
    """
    glyph_crop_start: Optional[Tuple[int, int]] = None
    glyph_crop_size: Optional[Tuple[int, int]] = None
    num_memory_units: int = 128
    memory_dim: int = 64
    use_bl_stats: bool = True
    bl_stats_log_transform: bool = False
    num_bl_stats_blocks: int = 2
    num_perceiver_blocks: int = 2
    num_perceiver_self_attention_subblocks: int = 2
    transformer_dropout: float = 0.1
    transformer_fc_inner_dim: int = 256
    transformer_gate: Optional[str] = None
    memory_update_num_heads: int = 8
    map_attention_num_heads: int = 2
    use_fixed_positional_embeddings: bool = False
    positional_embeddings_num_bands: int = 32
    positional_embeddings_max_freq: int = 80
    deterministic: Optional[bool] = None

    def setup(self):
        if self.glyph_crop_size is not None:
            self._glyphs_size = self.glyph_crop_size
        else:
            self._glyphs_size = nle.nethack.DUNGEON_SHAPE

        if self.use_fixed_positional_embeddings:
            self._glyph_pos_embedding_processor = nn.Dense(
                features=self.memory_dim,
                name='glyph_pos_embedding_processor',
            )
        else:
            self._glyph_pos_embedder = ItemEmbedder(
                num_items=self._glyphs_size[0] * self._glyphs_size[1],
                embedding_dim=self.memory_dim,
                name='glyph_pos_embedder',
            )

        self._glyph_embedder = nn.Embed(
            num_embeddings=nle.nethack.MAX_GLYPH + 1,
            features=self.memory_dim,
            name='glyph_embedder',
        )
        self._memory_embedder = ItemEmbedder(
            # bl_stats will add an extra memory unit
            num_items=self.num_memory_units - (1 if self.use_bl_stats else 0),
            embedding_dim=self.memory_dim,
            name='memory_embedder',
        )
        self._memory_update_blocks = [
            TransformerNet(
                num_blocks=self.num_perceiver_self_attention_subblocks,
                dim=self.memory_dim,
                fc_inner_dim=self.transformer_fc_inner_dim,
                gate=self.transformer_gate,
                num_heads=self.memory_update_num_heads,
                dropout_rate=self.transformer_dropout,
                deterministic=self.deterministic,
                name=f'perceiver_self_attention_block_{block_idx}',
            )
            for block_idx in range(self.num_perceiver_blocks)
        ]
        self._attention_to_prev_memory_block = CrossTransformerNet(
            num_blocks=1,
            dim=self.memory_dim,
            fc_inner_dim=self.transformer_fc_inner_dim,
            gate=self.transformer_gate,
            num_heads=self.memory_update_num_heads,
            dropout_rate=self.transformer_dropout,
            deterministic=self.deterministic,
            name='attention_to_prev_memory',
        )
        self._map_attention_blocks = [
            CrossTransformerNet(
                num_blocks=1,
                dim=self.memory_dim,
                fc_inner_dim=self.transformer_fc_inner_dim,
                gate=self.transformer_gate,
                num_heads=self.map_attention_num_heads,
                dropout_rate=self.transformer_dropout,
                deterministic=self.deterministic,
                name=f'perceiver_map_attention_block_{block_idx}',
            )
            for block_idx in range(self.num_perceiver_blocks)
        ]
        if self.use_bl_stats:
            self._bl_stats_network = DenseNet(
                num_blocks=self.num_bl_stats_blocks, dim=self.memory_dim, output_dim=self.memory_dim,
                name='bl_stats_network',
            )

    def _make_fixed_pos_embeddings(self):
        logf = jnp.linspace(
            start=0.0,
            stop=jnp.log(0.5 * self.positional_embeddings_max_freq),
            num=self.positional_embeddings_num_bands,
            dtype=jnp.float32,
        )
        f = jnp.exp(logf)

        r_coords = jnp.linspace(-1.0, 1.0, num=self._glyphs_size[0])
        c_coords = jnp.linspace(-1.0, 1.0, num=self._glyphs_size[1])
        x_2d, y_2d = jnp.meshgrid(r_coords, c_coords, indexing='ij')
        coords = jnp.stack([x_2d, y_2d], axis=-1)

        cfp = jnp.pi * jnp.einsum('...c,f->...cf', coords, f)
        cfp = jnp.reshape(
            cfp,
            (
                self._glyphs_size[0],
                self._glyphs_size[1],
                2 * self.positional_embeddings_num_bands
            )
        )
        sin_cfp = jnp.sin(cfp)
        cos_cfp = jnp.cos(cfp)

        pos_embeddings = jnp.concatenate([sin_cfp, cos_cfp, coords], axis=-1)
        pos_embeddings = self._glyph_pos_embedding_processor(pos_embeddings)
        pos_embeddings = jnp.reshape(pos_embeddings, (1, -1, self.memory_dim))

        return pos_embeddings

    def __call__(self, current_state_batch, prev_memory=None, deterministic=None):
        deterministic = nn.module.merge_param('deterministic', self.deterministic, deterministic)

        glyphs = current_state_batch['glyphs']
        batch_size = glyphs.shape[0]

        if self.glyph_crop_size is not None:
            # Can be used to crop unused observation area to speedup convergence
            if self.glyph_crop_start is not None:
                start_r, start_c = self.glyph_crop_start
            else:
                start_r = (nle.nethack.DUNGEON_SHAPE[0] - self.glyph_crop_size[0]) // 2
                start_c = (nle.nethack.DUNGEON_SHAPE[1] - self.glyph_crop_size[1]) // 2
            glyphs = glyphs[:, start_r:start_r + self.glyph_crop_size[0], start_c:start_c + self.glyph_crop_size[1]]

        # Initialize Perceiver latent memory embeddings
        memory = self._memory_embedder(batch_size)

        if self.use_bl_stats:
            bl_stats = current_state_batch['blstats']
            if self.bl_stats_log_transform:
                bl_stats = log_transform(bl_stats)
            bl_stats = self._bl_stats_network(bl_stats)
            # Add bl stats as an extra memory cell
            memory = jnp.concatenate([memory, jnp.expand_dims(bl_stats, axis=1)], axis=1)

        # If a memory context is specified, update memory by attending to it.
        # This allows for context-dependent state encoding
        if prev_memory is not None:
            memory = self._attention_to_prev_memory_block(memory, prev_memory, deterministic=deterministic)

        # Observed glyph embeddings
        glyphs = jnp.reshape(glyphs, newshape=(glyphs.shape[0], -1))
        glyphs_embeddings = self._glyph_embedder(glyphs)

        # Add positional embedding to glyphs (fixed or learned)
        if self.use_fixed_positional_embeddings:
            glyph_pos_embeddings = self._make_fixed_pos_embeddings()
        else:
            glyph_pos_embeddings = self._glyph_pos_embedder(batch_size)
        glyphs_embeddings += glyph_pos_embeddings

        # Perceiver body
        for block_idx in range(self.num_perceiver_blocks):
            memory = self._map_attention_blocks[block_idx](memory, glyphs_embeddings, deterministic=deterministic)
            memory = self._memory_update_blocks[block_idx](memory, deterministic=deterministic)

        return memory
