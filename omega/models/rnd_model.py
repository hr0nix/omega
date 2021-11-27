from dataclasses import field
from typing import Optional, Tuple, Dict

import flax.linen as nn
import jax.numpy as jnp
import jax.random

import nle.nethack

from omega.neural import TransformerNet, CrossTransformerNet


class StateEmbeddingModel(nn.Module):
    glyph_crop_area: Optional[Tuple[int, int]] = None
    glyph_embedding_dim: int = 32
    num_memory_units: int = 4
    memory_dim: int = 32
    output_dim: int = 128
    num_perceiver_blocks: int = 1
    num_perceiver_self_attention_subblocks: int = 1
    memory_update_num_heads: int = 2
    map_attention_num_heads: int = 2
    transformer_fc_inner_dim: int = 128
    transformer_dropout: float = 0.0  # Don't use any dropout to avoid rewards due to random network stochasticity
    deterministic: Optional[bool] = None

    def setup(self):
        if self.glyph_crop_area is not None:
            self._glyphs_size = self.glyph_crop_area
        else:
            self._glyphs_size = nle.nethack.DUNGEON_SHAPE

        self._glyph_pos_embedding = nn.Embed(
            num_embeddings=self._glyphs_size[0] * self._glyphs_size[1],
            features=self.glyph_embedding_dim
        )
        self._glyph_embedding = nn.Embed(
            num_embeddings=nle.nethack.MAX_GLYPH + 1,
            features=self.glyph_embedding_dim
        )
        self._memory_embedding = nn.Embed(
            num_embeddings=self.num_memory_units,
            features=self.memory_dim
        )
        self._memory_update_blocks = [
            TransformerNet(
                num_blocks=self.num_perceiver_self_attention_subblocks,
                dim=self.memory_dim,
                fc_inner_dim=self.transformer_fc_inner_dim,
                num_heads=self.memory_update_num_heads,
                dropout_rate=self.transformer_dropout,
                deterministic=self.deterministic,
                name='{}/perceiver_self_attention_block_{}'.format(self.name, block_idx)
            )
            for block_idx in range(self.num_perceiver_blocks)
        ]
        self._map_attention_blocks = [
            CrossTransformerNet(
                num_blocks=1,
                dim=self.memory_dim,
                fc_inner_dim=self.transformer_fc_inner_dim,
                num_heads=self.map_attention_num_heads,
                dropout_rate=self.transformer_dropout,
                deterministic=self.deterministic,
                name='{}/perceiver_map_attention_block_{}'.format(self.name, block_idx),
            )
            for block_idx in range(self.num_perceiver_blocks)
        ]
        self._output_network = nn.Dense(self.output_dim)

    def __call__(self, current_state_batch, rng, deterministic=None):
        deterministic = nn.module.merge_param('deterministic', self.deterministic, deterministic)

        glyphs = current_state_batch['glyphs']
        batch_size = glyphs.shape[0]

        if self.glyph_crop_area is not None:
            # Can be used to crop unused observation area to speedup convergence
            start_r = (nle.nethack.DUNGEON_SHAPE[0] - self.glyph_crop_area[0]) // 2
            start_c = (nle.nethack.DUNGEON_SHAPE[1] - self.glyph_crop_area[1]) // 2
            glyphs = glyphs[:, start_r:start_r + self.glyph_crop_area[0], start_c:start_c + self.glyph_crop_area[1]]

        # Perceiver latent memory embeddings
        memory_indices = jnp.arange(0, self.num_memory_units, dtype=jnp.int32)
        memory = self._memory_embedding(memory_indices)
        memory = jnp.tile(memory, reps=[batch_size, 1, 1])

        # Observed glyph embeddings
        glyphs = jnp.reshape(glyphs, newshape=(glyphs.shape[0], -1))
        glyphs_embeddings = self._glyph_embedding(glyphs)

        # Add positional embeddings
        glyph_pos_indices = jnp.arange(0, self._glyphs_size[0] * self._glyphs_size[1], dtype=jnp.int32)
        glyph_pos_embeddings = self._glyph_pos_embedding(glyph_pos_indices)
        glyphs_embeddings += glyph_pos_embeddings

        # Perceiver body
        for block_idx in range(self.num_perceiver_blocks):
            rng, subkey = jax.random.split(rng)
            memory = self._map_attention_blocks[block_idx](
                memory, glyphs_embeddings, deterministic=deterministic, rng=subkey)

            rng, subkey = jax.random.split(rng)
            memory = self._memory_update_blocks[block_idx](
                memory, deterministic=deterministic, rng=subkey)

        # Output
        memory = jnp.reshape(memory, newshape=[batch_size, -1])
        output = self._output_network(memory)

        return output


class RNDNetworkPair(nn.Module):
    state_embedding_model_config: Dict = field(default_factory=dict)
    deterministic: Optional[bool] = None

    def setup(self):
        self._predictor_network = StateEmbeddingModel(**self.state_embedding_model_config)
        self._random_network = StateEmbeddingModel(**self.state_embedding_model_config)

    def __call__(self, current_state_batch, rng, deterministic=None):
        deterministic = nn.module.merge_param('deterministic', self.deterministic, deterministic)

        subkey1, subkey2 = jax.random.split(rng)
        random_state = self._random_network(current_state_batch, rng=subkey1, deterministic=deterministic)
        random_state = jax.lax.stop_gradient(random_state)  # Don't train the random network
        predicted_state = self._predictor_network(current_state_batch, rng=subkey2, deterministic=deterministic)

        loss = 0.5 * (predicted_state - random_state) ** 2
        loss_per_example = jnp.mean(loss, axis=1)

        return loss_per_example
