import argparse
import yaml
import tqdm
import ray
import gym
import minihack
import wandb

import numpy as np

from absl import logging
logging.set_verbosity(logging.INFO)

from omega.agents import NethackMuZeroAgent
from omega.training import OnPolicyTrainer, ClusteringReplayBuffer
from omega.evaluation import EvaluationStats
from omega.minihack.rewards import distance_to_staircase_reward
from omega.utils.jax import disable_jit_if_no_gpu
from omega.utils.wandb import get_wandb_id


def make_env(train_config, episodes_dir):
    import omega.minihack.envs  # noqa

    reward_manager = None
    if train_config['use_dense_staircase_reward']:
        reward_manager = minihack.RewardManager()
        reward_manager.add_location_event('staircase', terminal_required=True)
        reward_manager.add_custom_reward_fn(distance_to_staircase_reward)

    return gym.make(
        train_config['env_name'],
        observation_keys=train_config['observation_keys'],
        savedir=episodes_dir,
        reward_manager=reward_manager,
    )


def load_config(filename):
    with open(filename, 'r') as f:
        return yaml.safe_load(f)


def main(args):
    config = load_config(args.config)
    train_config = config['train_config']

    ray.init(num_cpus=train_config['num_workers'])
    if args.wandb_id_file is not None:
        wandb.init(project='omega', config=config, resume='allow', id=get_wandb_id(args.wandb_id_file))

    env_factory = lambda: make_env(train_config, episodes_dir=args.episodes)
    env = env_factory()
    replay_buffer = ClusteringReplayBuffer(
        cluster_buffer_size=config['train_config']['replay_buffer_size'] // 2, num_clusters=2,
        # Uniform over good and bad trajectories
        clustering_fn=lambda t: 1 if np.sum(t['rewards']) >= config['train_config']['good_trajectory_reward_threshold'] else 0,
    )
    agent = NethackMuZeroAgent(
        replay_buffer=replay_buffer,
        observation_space=env.observation_space,
        action_space=env.action_space, config=config['agent_config'])
    trainer = OnPolicyTrainer(
        env_factory=env_factory,
        num_workers=train_config['num_workers'],
        num_envs=train_config['num_envs'],
        num_collection_steps=train_config['num_collection_steps'],
    )

    start_day = 0
    if args.checkpoints is not None:
        start_day = agent.try_load_from_checkpoint(args.checkpoints)
    logging.info('Starting from day {}'.format(start_day))

    stats = EvaluationStats()
    for day in tqdm.tqdm(range(start_day, train_config['num_days'])):
        trainer.run_training_step(agent, stats)

        if (day + 1) % train_config['epoch_every_num_days'] == 0:
            if args.wandb_id_file is not None:
                wandb.log(data=stats.to_dict(include_rolling_stats=True), step=day)
            stats.print_summary(title='After {} days:'.format(day + 1))
            if args.checkpoints is not None:
                agent.save_to_checkpoint(args.checkpoints, day)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', metavar='FILE', required=True)
    parser.add_argument('--checkpoints', metavar='DIR', required=False)
    parser.add_argument('--episodes', metavar='DIR', required=False)
    parser.add_argument('--wandb-id-file', metavar='FILE', required=False)
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    with disable_jit_if_no_gpu():
        main(args)
