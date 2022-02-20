# omega
This repo contains and implementation of an agent that can learn to maximise reward in environments with NetHack interface such as [nle](https://github.com/facebookresearch/nle) or [MiniHack](https://github.com/facebookresearch/minihack).

## Repo highlights:
* A [Perceiver](http://dpmd.ai/perceiver)-based encoder of NetHack states.
* An implementation of a [PPO](https://arxiv.org/abs/1707.06347)-based RL agent
  * Advantage is estimated using [GAE](https://arxiv.org/abs/1506.02438)
  * Per-batch advantage normalization and entropy-based policy regularization are supported.
* An implementation of [MuZero](https://arxiv.org/abs/1911.08265)-based RL agent.
  * MCTS runs on GPU and is pretty fast.
  * [Reanalyze](https://arxiv.org/abs/2104.06294) is supported.
  * State similarity loss loosely inspired by [EfficientZero](https://arxiv.org/abs/2111.00210) is supported.
* Training and inference is implemented in [JAX](https://github.com/google/jax), with the help of [rlax](https://github.com/deepmind/rlax) and [optax](https://github.com/deepmind/optax)
* Models are implemented in JAX/[Flax](https://github.com/google/flax)
