# omega
This repo contains and implementation of an agent that can learn to maximise reward in environments with NetHack interface such as [nle](https://github.com/facebookresearch/nle) or [MiniHack](https://github.com/facebookresearch/minihack).

![Crossing a river](/images/river.gif)
![Fighting monsters in a narrow corridor](/images/corridor.gif)

## Repo highlights
* A [Perceiver](http://dpmd.ai/perceiver)-inspired encoder of NetHack states.
* An implementation of a [PPO](https://arxiv.org/abs/1707.06347)-based RL agent
  * Advantage is estimated using [GAE](https://arxiv.org/abs/1506.02438)
  * Per-batch advantage normalization and entropy-based policy regularization are supported.
  * This agent was meant mainly as a baseline, most of the effort in this repo went into MuZero.
* An implementation of [MuZero](https://arxiv.org/abs/1911.08265)-based RL agent.
  * MCTS runs on GPU and is pretty fast.
  * [Reanalyze](https://arxiv.org/abs/2104.06294) is supported.
  * Recurrent memory is supported.
  * State consistency loss inspired by [Improving Model-Based Reinforcement Learning with Internal State Representations through Self-Supervision](https://arxiv.org/abs/2102.05599) is supported.
  * Ideas from [Stochastic MuZero](https://openreview.net/forum?id=X6D9bAHhBQ1) are implemented, so the agent runs correctly in stochastic environments.
  * A search policy from [Monte-Carlo tree search as regularized policy optimization](https://arxiv.org/pdf/2007.12509.pdf) can be enabled to improve efficiency of MCTS, which can be very helpful when simulation budget is small or branching factor is very large.
* Training and inference is implemented in [JAX](https://github.com/google/jax), with the help of [rlax](https://github.com/deepmind/rlax) and [optax](https://github.com/deepmind/optax)
* Models are implemented in JAX/[Flax](https://github.com/google/flax)

## How to train an agent (with docker)
1. Clone the repository:
```bash
git clone https://github.com/hr0nix/omega.git
```
2. Run the docker container:
```bash
bash ./omega/docker/run_container.sh
```
3. Create a new experiment based on one of the provided configs:
```bash
python3.8 ./tools/experiment_manager.py make --config ./configs/muzero/random_room_5x5.yaml --output-dir ./experiments/muzero_random_room_5x5
```
4. Run the newly created experiment. You can optionally track the experiment using [wandb](https://wandb.ai) (you will be asked if you want to, definitely recommended).
```bash
python3.8 ./tools/experiment_manager.py run --dir ./experiments/muzero_random_room_5x5 --gpu 0
```
5. After some episodes are completed, you can visualize them:
```bash
python3.8 ./tools/experiment_manager.py play --file ./experiments/muzero_random_room_5x5/episodes/<EPISODE_FILENAME_HERE>
```


## How to train an agent (with conda, without docker)
1. Create conda env
```bash
conda create -n omega python=3.8
conda activate omega
```
2. Clone the repository and install the omega module:
```bash
git clone https://github.com/hr0nix/omega.git
pip install -e "omega[cuda]"    # or cpu
```
3. Create a new experiment based on one of the provided configs:
```bash
python -m tools.experiment_manager make --config ./omega/configs/muzero/random_room_5x5.yaml --output-dir ./omega/experiments/muzero_random_room_5x5
```
4. Run the newly created experiment. You can optionally track the experiment using [wandb](https://wandb.ai) (you will be asked if you want to, definitely recommended).
```bash
python -m tools.experiment_manager run --dir ./omega/experiments/muzero_random_room_5x5 --gpu 0
```
5. After some episodes are completed, you can visualize them:
```bash
python -m tools.experiment_manager play --file ./omega/experiments/muzero_random_room_5x5/episodes/<EPISODE_FILENAME_HERE>
```
