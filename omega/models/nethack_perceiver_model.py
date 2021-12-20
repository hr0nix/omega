from dataclasses import field
from typing import Optional, Tuple, Dict

import flax.linen as nn
import jax.numpy as jnp
import jax.random

import nle.nethack

from omega.neural import TransformerNet, CrossTransformerNet, DenseNet


class ItemEmbedder(nn.Module):
    """
    Embeds a fixed number of items, replicates embeddings over batch size.
    """
    num_items: int
    embedding_dim: int

    def setup(self):
        self._embedder = nn.Embed(num_embeddings=self.num_items, features=self.embedding_dim)

    def __call__(self, batch_size):
        items = jnp.arange(0, self.num_items, dtype=jnp.int32)
        item_embeddings = self._embedder(items)
        item_embeddings = jnp.tile(item_embeddings, reps=[batch_size, 1, 1])

        return item_embeddings


class ItemSelector(nn.Module):
    """
    Given a set of item embeddings and some memory to attend to, computes item probabilities based on memory.
    """
    transformer_dim: int
    transformer_num_heads: int
    transformer_num_blocks: int
    transformer_fc_inner_dim: int
    transformer_dropout: float
    deterministic: Optional[bool] = None

    def setup(self):
        self._selection_transformer = CrossTransformerNet(
            num_blocks=self.transformer_num_blocks,
            dim=self.transformer_dim,
            fc_inner_dim=self.transformer_fc_inner_dim,
            num_heads=self.transformer_num_heads,
            dropout_rate=self.transformer_dropout,
            deterministic=self.deterministic,
            name='{}/selection_transformer'.format(self.name),
        )
        self._logits_producer = nn.Dense(1)

    def __call__(self, items, memory, rng, deterministic=None):
        deterministic = nn.module.merge_param('deterministic', self.deterministic, deterministic)

        rng, subkey = jax.random.split(rng)

        items = self._selection_transformer(items, memory, deterministic=deterministic, rng=subkey)
        item_logits = self._logits_producer(items)
        item_logits = jnp.squeeze(item_logits, axis=-1)
        item_log_probs = jax.nn.log_softmax(item_logits, axis=-1)

        return item_log_probs


class PerceiverStateEncoder(nn.Module):
    """
    Encodes a nethack state observation into a latent memory vector.
    """
    glyph_crop_area: Optional[Tuple[int, int]] = None
    glyph_embedding_dim: int = 64
    num_memory_units: int = 128
    memory_dim: int = 64
    use_bl_stats: bool = True
    num_bl_stats_blocks: int = 2
    num_perceiver_blocks: int = 2
    num_perceiver_self_attention_subblocks: int = 2
    transformer_dropout: float = 0.1
    transformer_fc_inner_dim: int = 256
    memory_update_num_heads: int = 8
    map_attention_num_heads: int = 2
    use_fixed_positional_embeddings: bool = False
    positional_embeddings_num_bands: int = 32
    positional_embeddings_max_freq: int = 80
    deterministic: Optional[bool] = None

    def setup(self):
        if self.glyph_crop_area is not None:
            self._glyphs_size = self.glyph_crop_area
        else:
            self._glyphs_size = nle.nethack.DUNGEON_SHAPE

        if self.use_fixed_positional_embeddings:
            self._glyph_pos_embedding_processor = nn.Dense(self.memory_dim)
        else:
            self._glyph_pos_embedding = ItemEmbedder(
                num_items=self._glyphs_size[0] * self._glyphs_size[1],
                embedding_dim=self.glyph_embedding_dim
            )

        self._glyph_embedding = nn.Embed(
            num_embeddings=nle.nethack.MAX_GLYPH + 1,
            features=self.glyph_embedding_dim
        )
        self._memory_embedding = ItemEmbedder(
            num_items=self.num_memory_units,
            embedding_dim=self.memory_dim
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
        if self.use_bl_stats:
            self._bl_stats_network = DenseNet(
                num_blocks=self.num_bl_stats_blocks, dim=self.memory_dim, output_dim=self.memory_dim,
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
        memory = self._memory_embedding(batch_size)

        if self.use_bl_stats:
            bl_stats = current_state_batch['blstats']
            bl_stats = self._bl_stats_network(bl_stats)
            memory = memory + jnp.expand_dims(bl_stats, axis=1)  # Add global features to every memory cell

        # Observed glyph embeddings
        glyphs = jnp.reshape(glyphs, newshape=(glyphs.shape[0], -1))
        glyphs_embeddings = self._glyph_embedding(glyphs)

        # Add positional embedding to glyphs (fixed or learned)
        if self.use_fixed_positional_embeddings:
            glyph_pos_embeddings = self._make_fixed_pos_embeddings()
        else:
            glyph_pos_embeddings = self._glyph_pos_embedding(batch_size)
        glyphs_embeddings += glyph_pos_embeddings

        # Perceiver body
        for block_idx in range(self.num_perceiver_blocks):
            rng, subkey1, subkey2 = jax.random.split(rng, 3)
            memory = self._map_attention_blocks[block_idx](
                memory, glyphs_embeddings, deterministic=deterministic, rng=subkey1)
            memory = self._memory_update_blocks[block_idx](
                memory, deterministic=deterministic, rng=subkey2)

        return memory


class NethackPerceiverModel(nn.Module):
    num_actions: int
    state_encoder_config: Dict = field(default_factory=dict)
    output_attention_num_heads: int = 8
    transformer_dropout: float = 0.1
    transformer_fc_inner_dim: int = 256
    num_policy_network_heads: int = 2
    num_policy_network_blocks: int = 1
    num_inverse_dynamics_network_heads: int = 2
    num_inverse_dynamics_network_blocks: int = 1
    num_value_network_blocks: int = 2
    deterministic: Optional[bool] = None

    def setup(self):
        self._state_encoder = PerceiverStateEncoder(**self.state_encoder_config)
        self._output_embedding = ItemEmbedder(
            num_items=1,  # state value
            embedding_dim=self._state_encoder.memory_dim
        )
        self._action_embedding = ItemEmbedder(
            num_items=self.num_actions,
            embedding_dim=self._state_encoder.memory_dim
        )
        self._output_transformer = CrossTransformerNet(
            num_blocks=1,
            dim=self._state_encoder.memory_dim,
            fc_inner_dim=self.transformer_fc_inner_dim,
            num_heads=self.output_attention_num_heads,
            dropout_rate=self.transformer_dropout,
            deterministic=self.deterministic,
            name='{}/output_transformer'.format(self.name),
        )
        self._policy_network = ItemSelector(
            transformer_dim=self._state_encoder.memory_dim,
            transformer_num_blocks=self.num_policy_network_blocks,
            transformer_num_heads=self.num_policy_network_heads,
            transformer_fc_inner_dim=self.transformer_fc_inner_dim,
            transformer_dropout=self.transformer_dropout,
            deterministic=self.deterministic,
        )
        self._inverse_dynamics_model = ItemSelector(
            transformer_dim=self._state_encoder.memory_dim,
            transformer_num_blocks=self.num_inverse_dynamics_network_blocks,
            transformer_num_heads=self.num_inverse_dynamics_network_heads,
            transformer_fc_inner_dim=self.transformer_fc_inner_dim,
            transformer_dropout=self.transformer_dropout,
            deterministic=self.deterministic,
        )
        self._value_network = DenseNet(
            num_blocks=self.num_value_network_blocks, dim=self._state_encoder.memory_dim, output_dim=1,
        )

    def __call__(self, current_state, next_state, rng, deterministic=None):
        deterministic = nn.module.merge_param('deterministic', self.deterministic, deterministic)

        rng, subkey = jax.random.split(rng)
        memory = self._state_encoder(current_state, deterministic=deterministic, rng=subkey)
        batch_size = memory.shape[0]

        # Attend to latent memory from each regular output
        rng, subkey = jax.random.split(rng)
        output_embeddings = self._output_embedding(batch_size)
        output_embeddings = self._output_transformer(
            output_embeddings, memory, deterministic=deterministic, rng=subkey)

        # Compute state values
        state_value = self._value_network(output_embeddings[:, 0, :])
        state_value = jnp.squeeze(state_value, axis=-1)

        # Embed actions (for policy network and inverse dynamics model)
        action_embeddings = self._action_embedding(batch_size)

        # Compute action probs
        rng, subkey = jax.random.split(rng)
        log_action_probs = self._policy_network(
            action_embeddings, memory, deterministic=deterministic, rng=subkey)

        # Model inverse dynamics: predict the action that transitions into the next state
        rng, subkey1, subkey2 = jax.random.split(rng, 3)
        next_state_memory = self._state_encoder(next_state, deterministic=deterministic, rng=subkey1)
        combined_memory = jnp.concatenate([memory, next_state_memory], axis=1)
        action_embeddings = jax.lax.stop_gradient(action_embeddings)  # Do not update action embeddings
        log_id_action_probs = self._inverse_dynamics_model(
            action_embeddings, combined_memory, deterministic=deterministic, rng=subkey2)

        return log_action_probs, log_id_action_probs, state_value
