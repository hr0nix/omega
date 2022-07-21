import argparse
import yaml
import tqdm
import ray
import gym
import minihack  # noqa
import wandb
import jax

from omega.agents import NethackMuZeroAgent, NethackPPOAgent
from omega.training import OnPolicyTrainer, DummyTrainer
from omega.training.replay_buffer import create_from_config as create_replay_buffer_from_config
from omega.evaluation import EvaluationStats
from omega.utils.profiling import enable_profiling
from omega.utils.gym import NetHackRGBRendering, NetHackBLStatsFiltering
from omega.utils.jax import conditionally_disable_jit
from omega.utils.wandb import get_wandb_id

from absl import logging
logging.set_verbosity(logging.INFO)


def make_env(train_config, episodes_dir, episode_video_dir):
    # Import custom environments defined in omega
    import omega.minihack.envs  # noqa

    observation_keys = train_config['observation_keys']
    if episode_video_dir is not None and 'pixel' not in observation_keys:
        observation_keys.append('pixel')

    env = gym.make(
        train_config['env_name'],
        observation_keys=observation_keys,
        savedir=episodes_dir,
        disable_env_checker=True,
    )

    if episode_video_dir is not None:
        env = NetHackRGBRendering(env, episode_video_dir)

    if train_config.get('filter_bl_stats') or train_config.get('keep_bl_stats'):
        env = NetHackBLStatsFiltering(
            env, keys_to_filter=train_config.get('filter_bl_stats'), keys_to_keep=train_config.get('keep_bl_stats'))

    return env


def load_config(filename):
    with open(filename, 'r') as f:
        return yaml.safe_load(f)


def create_agent(config, env):
    agent_type = config['train_config']['agent_type']
    if agent_type == 'muzero':
        replay_buffer = create_replay_buffer_from_config(config['train_config']['replay_buffer'])
        return NethackMuZeroAgent(
            replay_buffer=replay_buffer,
            observation_space=env.observation_space,
            action_space=env.action_space,
            config=config['agent_config']
        )
    elif agent_type == 'ppo':
        return NethackPPOAgent(
            observation_space=env.observation_space,
            action_space=env.action_space,
            config=config['agent_config']
        )
    else:
        raise ValueError(f'Unknown agent type: {agent_type}')


def train_agent(args):
    with conditionally_disable_jit(args.disable_jit):
        if args.log_profile:
            logging.info('Profiling is enabled')
            enable_profiling()

        if args.log_memory_transfer:
            logging.info('Memory transfer logging is enabled')
            jax.config.update('jax_transfer_guard', 'log')

        if args.log_compilation:
            logging.info('JIT compilation logging is enabled')
            jax.config.update('jax_log_compiles', True)

        config = load_config(args.config)
        train_config = config['train_config']

        ray.init(num_cpus=train_config['num_workers'], local_mode=args.ray_local_mode)
        if args.wandb_id_file is not None:
            wandb.init(project='omega', config=config, resume='allow', id=get_wandb_id(args.wandb_id_file))

        def env_factory():
            return make_env(train_config, episodes_dir=args.episodes, episode_video_dir=args.episode_videos)
        env = env_factory()
        agent = create_agent(config, env)
        trainer = OnPolicyTrainer(
            agent=agent,
            env_factory=env_factory,
            num_workers=train_config['num_workers'],
            num_envs=train_config['num_envs'],
            num_collection_steps=train_config['num_collection_steps'],
            allow_to_act_in_terminal_state_once=train_config['allow_to_act_in_terminal_state_once'],
        )

        start_day = 0
        if args.checkpoints is not None:
            start_day = agent.try_load_from_checkpoint(args.checkpoints)
        logging.info('Starting from day {}'.format(start_day))

        stats = EvaluationStats(discount_factor=config['agent_config']['discount_factor'])
        for day in tqdm.tqdm(range(start_day, train_config['num_days'])):
            trainer.run_training_step(stats)

            if (day + 1) % train_config['epoch_every_num_days'] == 0:
                if args.wandb_id_file is not None:
                    wandb.log(data=stats.to_dict(include_rolling_stats=True), step=day, commit=True)
                stats.print_summary(title='After {} days:'.format(day + 1))
                if args.checkpoints is not None:
                    agent.save_to_checkpoint(args.checkpoints)


def eval_agent(args):
    ray.init(num_cpus=args.num_parallel_envs)

    config = load_config(args.train_config)

    def env_factory():
        return make_env(config['train_config'], episodes_dir=args.episodes, episode_video_dir=args.episode_videos)
    env = env_factory()
    agent = create_agent(config, env)
    trainer = DummyTrainer(
        agent=agent,
        env_factory=env_factory,
        num_workers=args.num_workers,
        num_envs=args.num_parallel_envs,
        num_collection_steps=1,
    )

    start_day = agent.try_load_from_checkpoint(args.checkpoints)
    if start_day == 0:
        raise RuntimeError('Looks like no checkpoint has been found')

    stats = EvaluationStats(discount_factor=config['agent_config']['discount_factor'])
    for _ in tqdm.tqdm(range(args.num_steps)):
        trainer.run_training_step(stats)

    stats.print_summary(title='Stats after taking {} steps:'.format(args.num_steps))


def parse_args():
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(required=True, dest='mode', help='Available modes:')

    train_parser = subparsers.add_parser('train', help='Train an agent')
    train_parser.add_argument('--config', metavar='FILE', required=True)
    train_parser.add_argument('--checkpoints', metavar='DIR', required=False)
    train_parser.add_argument('--episodes', metavar='DIR', required=False)
    train_parser.add_argument('--episode-videos', metavar='DIR', required=False)
    train_parser.add_argument('--wandb-id-file', metavar='FILE', required=False)
    train_parser.add_argument('--log-memory-transfer', action='store_true', required=False, default=False)
    train_parser.add_argument('--log-profile', action='store_true', required=False, default=False)
    train_parser.add_argument('--log-compilation', action='store_true', required=False, default=False)
    train_parser.add_argument('--disable-jit', action='store_true', required=False, default=False)
    train_parser.add_argument('--ray-local-mode', action='store_true', required=False, default=False)
    train_parser.set_defaults(func=train_agent)

    eval_parser = subparsers.add_parser('eval', help='Eval an agent')
    eval_parser.add_argument('--train-config', metavar='FILE', required=True)
    eval_parser.add_argument('--checkpoints', metavar='DIR', required=True)
    eval_parser.add_argument('--episodes', metavar='DIR', required=False)
    eval_parser.add_argument('--episode-videos', metavar='DIR', required=False)
    eval_parser.add_argument('--num-steps', metavar='NUM', type=int, default=500, required=False)
    eval_parser.add_argument('--num-workers', metavar='NUM', type=int, default=2, required=False)
    eval_parser.add_argument('--num-parallel-envs', metavar='NUM', type=int, default=32, required=False)
    eval_parser.set_defaults(func=eval_agent)

    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    args.func(args)
