from typing import Optional

import flax.linen as nn
import jax.numpy as jnp
import jax.random

from omega.neural import CrossTransformerNet


class ItemEmbedder(nn.Module):
    """
    Embeds a fixed number of items, replicates embeddings over batch size.
    """
    num_items: int
    embedding_dim: int

    def setup(self):
        self._embedder = nn.Embed(
            num_embeddings=self.num_items, features=self.embedding_dim,
            name='embedder'
        )

    def __call__(self, batch_size):
        items = jnp.arange(0, self.num_items, dtype=jnp.int32)
        item_embeddings = self._embedder(items)
        item_embeddings = jnp.tile(item_embeddings, reps=(batch_size, 1, 1))

        return item_embeddings


class ItemSelector(nn.Module):
    """
    Given a set of item embeddings and some memory to attend to, computes item probabilities based on memory.
    """
    transformer_dim: int
    transformer_num_blocks: int
    transformer_fc_inner_dim: int
    transformer_num_heads: int = 1
    transformer_dropout: float = 0.1
    deterministic: Optional[bool] = None

    def setup(self):
        self._selection_transformer = CrossTransformerNet(
            num_blocks=self.transformer_num_blocks,
            dim=self.transformer_dim,
            fc_inner_dim=self.transformer_fc_inner_dim,
            num_heads=self.transformer_num_heads,
            dropout_rate=self.transformer_dropout,
            deterministic=self.deterministic,
            name='selection_transformer'.format(self.name),
        )
        self._logits_producer = nn.Dense(
            features=1,
            name='logits_producer'
        )

    def __call__(self, items, memory, rng, deterministic=None):
        deterministic = nn.module.merge_param('deterministic', self.deterministic, deterministic)

        rng, subkey = jax.random.split(rng)

        items = self._selection_transformer(items, memory, deterministic=deterministic, rng=subkey)
        item_logits = self._logits_producer(items)
        item_logits = jnp.squeeze(item_logits, axis=-1)
        item_log_probs = jax.nn.log_softmax(item_logits, axis=-1)

        return item_log_probs


class ItemPredictor(nn.Module):
    """
    Given memory and a number of output positions, computes the outputs using attention from output position embeddings.
    """
    num_outputs: int
    transformer_dim: int
    transformer_num_blocks: int
    transformer_fc_inner_dim: int
    transformer_num_heads: int = 1
    transformer_dropout: float = 0.1
    deterministic: Optional[bool] = None

    def setup(self):
        self._output_transformer = CrossTransformerNet(
            num_blocks=self.transformer_num_blocks,
            dim=self.transformer_dim,
            fc_inner_dim=self.transformer_fc_inner_dim,
            num_heads=self.transformer_num_heads,
            dropout_rate=self.transformer_dropout,
            deterministic=self.deterministic,
            name='selection_transformer',
        )
        self._output_embedder = ItemEmbedder(
            num_items=self.num_outputs, embedding_dim=self.transformer_dim,
            name='output_embedder'
        )
        self._scalar_producer = nn.Dense(
            features=1,
            name='scalar_producer',
        )

    def __call__(self, memory, rng, deterministic=None):
        deterministic = nn.module.merge_param('deterministic', self.deterministic, deterministic)
        rng, subkey = jax.random.split(rng)

        output_embeddings = self._output_embedder(batch_size=memory.shape[0])
        output_embeddings = self._output_transformer(
            output_embeddings, memory, deterministic=deterministic, rng=subkey)
        outputs = self._scalar_producer(output_embeddings)
        outputs = jnp.squeeze(outputs, -1)

        return outputs
