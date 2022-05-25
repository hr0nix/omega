import argparse
import yaml
import tqdm
import ray
import gym
import minihack
import wandb

from absl import logging
logging.set_verbosity(logging.INFO)

from omega.agents import NethackMuZeroAgent
from omega.training import OnPolicyTrainer, DummyTrainer
from omega.training.replay_buffer import create_from_config as create_replay_buffer_from_config
from omega.evaluation import EvaluationStats
from omega.utils.jax import disable_jit_if_no_gpu
from omega.utils.wandb import get_wandb_id


def make_env(train_config, episodes_dir):
    # Import custom environments defined in omega
    import omega.minihack.envs  # noqa

    return gym.make(
        train_config['env_name'],
        observation_keys=train_config['observation_keys'],
        savedir=episodes_dir,
    )


def load_config(filename):
    with open(filename, 'r') as f:
        return yaml.safe_load(f)


def train_agent(args):
    config = load_config(args.config)
    train_config = config['train_config']

    ray.init(num_cpus=train_config['num_workers'])
    if args.wandb_id_file is not None:
        wandb.init(project='omega', config=config, resume='allow', id=get_wandb_id(args.wandb_id_file))

    env_factory = lambda: make_env(train_config, episodes_dir=args.episodes)
    env = env_factory()
    replay_buffer = create_replay_buffer_from_config(config['train_config']['replay_buffer'])
    agent = NethackMuZeroAgent(
        replay_buffer=replay_buffer,
        observation_space=env.observation_space,
        action_space=env.action_space, config=config['agent_config'])
    trainer = OnPolicyTrainer(
        agent=agent,
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
        trainer.run_training_step(stats)

        if (day + 1) % train_config['epoch_every_num_days'] == 0:
            if args.wandb_id_file is not None:
                wandb.log(data=stats.to_dict(include_rolling_stats=True), step=day)
            stats.print_summary(title='After {} days:'.format(day + 1))
            if args.checkpoints is not None:
                agent.save_to_checkpoint(args.checkpoints, day)


def eval_agent(args):
    config = load_config(args.train_config)
    ray.init(num_cpus=args.num_parallel_envs)

    env_factory = lambda: make_env(config['train_config'], episodes_dir=args.episodes)
    env = env_factory()
    agent = NethackMuZeroAgent(
        replay_buffer=None,
        observation_space=env.observation_space,
        action_space=env.action_space, config=config['agent_config'])
    trainer = DummyTrainer(
        env_factory=env_factory,
        num_workers=args.num_workers,
        num_envs=args.num_parallel_envs,
        num_collection_steps=1,
    )

    start_day = agent.try_load_from_checkpoint(args.checkpoints)
    if start_day == 0:
        raise RuntimeError('Looks like no checkpoint has been found')

    stats = EvaluationStats()
    for _ in tqdm.tqdm(range(args.num_steps)):
        trainer.run_training_step(agent, stats)

    stats.print_summary(title='Stats after taking {} steps:'.format(args.num_steps))


def parse_args():
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(required=True, dest='mode', help='Available modes:')

    train_parser = subparsers.add_parser('train', help='Train an agent')
    train_parser.add_argument('--config', metavar='FILE', required=True)
    train_parser.add_argument('--checkpoints', metavar='DIR', required=False)
    train_parser.add_argument('--episodes', metavar='DIR', required=False)
    train_parser.add_argument('--wandb-id-file', metavar='FILE', required=False)
    train_parser.set_defaults(func=train_agent)

    eval_parser = subparsers.add_parser('eval', help='Eval an agent')
    eval_parser.add_argument('--train-config', metavar='FILE', required=True)
    eval_parser.add_argument('--checkpoints', metavar='DIR', required=True)
    eval_parser.add_argument('--episodes', metavar='DIR', required=False)
    eval_parser.add_argument('--num-steps', metavar='NUM', type=int, default=500, required=False)
    eval_parser.add_argument('--num-workers', metavar='NUM', type=int, default=2, required=False)
    eval_parser.add_argument('--num-parallel-envs', metavar='NUM', type=int, default=32, required=False)
    eval_parser.set_defaults(func=eval_agent)

    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    with disable_jit_if_no_gpu():
        args.func(args)