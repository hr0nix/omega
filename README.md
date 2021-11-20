# omega
This repo contains and implementation of an agent that can learn to maximise reward in environments with NetHack interface such as [nle](https://github.com/facebookresearch/nle) or [MiniHack](https://github.com/facebookresearch/minihack).

## Repo highlights:
* An implementation of a [PPO](https://arxiv.org/abs/1707.06347)-based training agent
  * Advantage is estimated using [GAE](https://arxiv.org/abs/1506.02438)
  * Per-batch advantage normalization is supported.
  * Entropy-based policy regularization is supported.
* A [Perceiver](http://dpmd.ai/perceiver)-based model of policy and value functions.
* Training and inference is implemented in [JAX](https://github.com/google/jax), with the help of [rlax](https://github.com/deepmind/rlax) and [optax](https://github.com/deepmind/optax)
* Models are implemented in JAX/[Flax](https://github.com/google/flax)
