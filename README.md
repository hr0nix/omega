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

## How to try it:
First, clone the repository:

`$ git clone https://github.com/hr0nix/omega.git`

Now run the docker container:

`$ bash ./omega/docker/run_container.sh`

Create a new experiment based on one of the provided configs:

`./tools/experiment_manager.py make --config ./configs/muzero/random_room_5x5.yaml --output-dir ./experiments/muzero_random_room_5x5`

You can now run the newly created experiment:

`$ ./tools/experiment_manager.py run --dir ./experiments/muzero_random_room_5x5 --gpu 0`

After some episodes are completed, you can visualize them:

`$ ./tools/experiment_manager.py play --file ./experiments/muzero_random_room_5x5/episodes/<EPISODE_FILENAME_HERE>`

You can also track the experiment using [wandb](https://wandb.ai) (you will be asked if you want to when the training is being started).

