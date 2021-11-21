from typing import Optional, Tuple

import flax.linen as nn
import jax.numpy as jnp
import jax.random

import nle.nethack

from omega.neural import TransformerNet, CrossTransformerNet, DenseNet


class NethackPerceiverModel(nn.Module):
    num_actions: int
    glyph_crop_area: Optional[Tuple[int, int]] = None
    glyph_embedding_dim: int = 64
    output_attention_num_heads: int = 8
    num_memory_units: int = 128
    memory_dim: int = 64
    use_bl_stats: bool = True
    num_bl_stats_blocks: int = 2
    num_perceiver_blocks: int = 2
    num_perceiver_self_attention_subblocks: int = 2
    transformer_dropout: float = 0.1
    memory_update_num_heads: int = 8
    map_attention_num_heads: int = 2
    transformer_fc_inner_dim: int = 256
    num_policy_network_blocks: int = 4
    num_value_network_blocks: int = 4
    deterministic: Optional[bool] = None
    use_fixed_positional_embeddings: bool = False
    positional_embeddings_num_bands: int = 32
    positional_embeddings_max_freq: int = 80

    def setup(self):
        if self.glyph_crop_area is not None:
            self._glyphs_size = self.glyph_crop_area
        else:
            self._glyphs_size = nle.nethack.DUNGEON_SHAPE

        if not self.use_fixed_positional_embeddings:
            self._glyph_pos_embedding = nn.Embed(
                num_embeddings=self._glyphs_size[0] * self._glyphs_size[1],
                features=self.glyph_embedding_dim
            )
        else:
            self._glyph_pos_embedding_processor = nn.Dense(self.memory_dim)
        self._glyph_embedding = nn.Embed(
            num_embeddings=nle.nethack.MAX_GLYPH + 1,
            features=self.glyph_embedding_dim
        )
        self._memory_embedding = nn.Embed(
            num_embeddings=self.num_memory_units,
            features=self.memory_dim
        )
        self._output_embedding = nn.Embed(
            num_embeddings=2,
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
        self._output_transformer = CrossTransformerNet(
                num_blocks=1,
                dim=self.memory_dim,
                fc_inner_dim=self.transformer_fc_inner_dim,
                num_heads=self.output_attention_num_heads,
                dropout_rate=self.transformer_dropout,
                deterministic=self.deterministic,
                name='{}/output_attention'.format(self.name),
            )
        if self.use_bl_stats:
            self._bl_stats_network = DenseNet(
                num_blocks=self.num_bl_stats_blocks, dim=self.memory_dim, output_dim=self.memory_dim,
            )
        self._policy_network = DenseNet(
            num_blocks=self.num_policy_network_blocks, dim=self.memory_dim, output_dim=self.num_actions,
        )
        self._value_network = DenseNet(
            num_blocks=self.num_value_network_blocks, dim=self.memory_dim, output_dim=1,
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

    def __call__(self, observations_batch, rng, deterministic=None):
        deterministic = nn.module.merge_param('deterministic', self.deterministic, deterministic)

        glyphs = observations_batch['glyphs']
        batch_size = glyphs.shape[0]

        if self.glyph_crop_area is not None:
            # Can be used to crop unused observation area to speedup convergence
            start_r = (nle.nethack.DUNGEON_SHAPE[0] - self.glyph_crop_area[0]) // 2
            start_c = (nle.nethack.DUNGEON_SHAPE[1] - self.glyph_crop_area[1]) // 2
            glyphs = glyphs[:, start_r:start_r + self.glyph_crop_area[0], start_c:start_c + self.glyph_crop_area[1]]

        # Perceiver latent memory embeddings
        memory_indices = jnp.arange(0, self.num_memory_units, dtype=jnp.int32)
        memory = self._memory_embedding(memory_indices)  # TODO: add memory recurrence
        memory = jnp.tile(memory, reps=[batch_size, 1, 1])

        if self.use_bl_stats:
            bl_stats = observations_batch['blstats']
            bl_stats = self._bl_stats_network(bl_stats)
            memory = memory + jnp.expand_dims(bl_stats, axis=1)  # Add global features to every memory cell

        # Observed glyph embeddings
        glyphs = jnp.reshape(glyphs, newshape=(glyphs.shape[0], -1))
        glyphs_embeddings = self._glyph_embedding(glyphs)

        # Add positional embedding to glyphs (fixed or learned)
        if self.use_fixed_positional_embeddings:
            glyph_pos_embeddings = self._make_fixed_pos_embeddings()
        else:
            glyph_pos_indices = jnp.arange(
                0, self._glyphs_size[0] * self._glyphs_size[1], dtype=jnp.int32)
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

        # Attend to latent memory from each output
        rng, subkey = jax.random.split(rng)
        output_indices = jnp.arange(0, 2, dtype=jnp.int32)
        output_embeddings = self._output_embedding(output_indices)
        output_embeddings = jnp.tile(output_embeddings, reps=[batch_size, 1, 1])
        output_embeddings = self._output_transformer(
            output_embeddings, memory, deterministic=deterministic, rng=subkey)

        # Compute action probs
        action_logits = self._policy_network(output_embeddings[:, 0, :])
        action_logprobs = jax.nn.log_softmax(action_logits, axis=-1)

        # Compute state values
        state_value = self._value_network(output_embeddings[:, 1, :])
        state_value = jnp.squeeze(state_value, axis=-1)

        return action_logprobs, state_value
