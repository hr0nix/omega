import argparse
import yaml
import tqdm

import gym
import nle

import clu.metric_writers

from absl import logging
logging.set_verbosity(logging.INFO)

from omega.agents import RandomAgent, NethackTransformerAgent
from omega.training import OnPolicyTrainer
from omega.evaluation import EvaluationStats
from omega.utils.jax import disable_jit_if_no_gpu


def env_factory():
    return gym.make("NetHackScore-v0", observation_keys=['glyphs', 'blstats'])


def load_config(filename):
    with open(filename, 'r') as f:
        return yaml.safe_load(f)


def main(args):
    config = load_config(args.config)

    env = env_factory()
    agent = NethackTransformerAgent(env.observation_space, env.action_space, config=config['model_config'])
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
                log_writer.write_scalars(day, stats.to_dict())
            stats.print_summary(title='After {} days:'.format(day + 1))
            stats.reset()

            if args.checkpoints is not None:
                agent.save_to_checkpoint(args.checkpoints, day)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', metavar='FILE', required=True)
    parser.add_argument('--tb-logs', metavar='DIR', required=False)
    parser.add_argument('--checkpoints', metavar='DIR', required=False)
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    with disable_jit_if_no_gpu():
        main(args)
