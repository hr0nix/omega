import argparse
import yaml
import tqdm

import gym
import minihack

import clu.metric_writers

from absl import logging
logging.set_verbosity(logging.INFO)

from omega.agents import NethackPPOAgent
from omega.models import NethackPerceiverModel
from omega.training import OnPolicyTrainer
from omega.evaluation import EvaluationStats
from omega.minihack.rewards import distance_to_staircase_reward
from omega.utils.jax import disable_jit_if_no_gpu


def make_env(game_logs_dir, use_dense_reward):
    reward_manager = None
    if use_dense_reward:
        reward_manager = minihack.RewardManager()
        reward_manager.add_location_event('staircase', terminal_required=True)
        reward_manager.add_custom_reward_fn(distance_to_staircase_reward)

    return gym.make(
        'MiniHack-Room-Random-5x5-v0',
        observation_keys=['glyphs', 'blstats'],
        savedir=game_logs_dir,
        reward_manager=reward_manager,
    )


def load_config(filename):
    with open(filename, 'r') as f:
        return yaml.safe_load(f)


def main(args):
    config = load_config(args.config)

    env_factory = lambda: make_env(game_logs_dir=args.game_logs, use_dense_reward=config['use_dense_reward'])
    env = env_factory()
    agent = NethackPPOAgent(
        NethackPerceiverModel, env.observation_space, env.action_space, config=config['agent_config'])
    trainer = OnPolicyTrainer(
        env_factory=env_factory,
        num_parallel_envs=config['num_parallel_envs'],
        num_collection_steps=config['num_collection_steps'],
    )

    start_day = 0
    if args.checkpoints is not None:
        start_day = agent.try_load_from_checkpoint(args.checkpoints)
    logging.info('Starting from day {}'.format(start_day))

    log_writer = None
    if args.tb_logs is not None:
        log_writer = clu.metric_writers.create_default_writer(args.tb_logs)

    stats = EvaluationStats()
    for day in tqdm.tqdm(range(start_day, config['num_days'])):
        trainer.run_training_step(agent, stats)

        if (day + 1) % config['epoch_every_num_days'] == 0:
            if log_writer is not None:
                log_writer.write_scalars(
                    day,
                    stats.to_dict(include_rolling_stats=True)
                )
            stats.print_summary(title='After {} days:'.format(day + 1))

            if args.checkpoints is not None:
                agent.save_to_checkpoint(args.checkpoints, day)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', metavar='FILE', required=True)
    parser.add_argument('--tb-logs', metavar='DIR', required=False)
    parser.add_argument('--checkpoints', metavar='DIR', required=False)
    parser.add_argument('--game-logs', metavar='DIR', required=False)
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    with disable_jit_if_no_gpu():
        main(args)
