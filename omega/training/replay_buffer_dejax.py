import jax
import jax.numpy as jnp
import dejax

from ..utils.collections import get_dict_slice


def create_from_config(config):
    buffer_type = config['type']

    if buffer_type == 'fifo':
        return dejax.uniform_replay(max_size=config['buffer_size'])

    elif buffer_type == 'uniform_over_good_and_bad':
        rewards_field = config.get('rewards_field', 'rewards')

        def clustering_fn(trajectory):
            return jax.lax.select(
                jnp.sum(trajectory[rewards_field]) >= config['good_total_reward_threshold'],
                on_true=1,
                on_false=0,
            )

        return dejax.clustered_replay(
            cluster_buffer=create_from_config(config['cluster_buffer']),
            num_clusters=2,
            clustering_fn=clustering_fn,
            **get_dict_slice(config, ['distribution_power'])
        )

    else:
        raise ValueError(f'Unknown replay buffer type: {buffer_type}')
