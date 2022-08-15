import argparse
import shutil
import os
import subprocess

from absl import logging
logging.set_verbosity(logging.INFO)


CONFIG_FILENAME = 'config.yaml'
CHECKPOINTS_DIR = 'checkpoints'
EPISODES_DIR = 'episodes'
WANDB_ID_FILE = 'wandb_id'


def make_experiment(args):
    if os.path.exists(args.output_dir):
        raise RuntimeError(f'Directory {args.output_dir} already exists')
    if not os.path.exists(args.config):
        raise RuntimeError(f'Cannot find config at {args.config}')
    if not os.path.isfile(args.config):
        raise RuntimeError(f'Config {args.config} is not a file')

    os.makedirs(args.output_dir, exist_ok=False)
    shutil.copy(args.config, os.path.join(args.output_dir, CONFIG_FILENAME))

    logging.info(f'A new experiment created at {args.output_dir}')


def run_experiment(args):
    if not os.path.exists(args.dir):
        raise RuntimeError(f'Directory {args.dir} not found')

    cur_dir = os.path.dirname(os.path.abspath(__file__))
    env = os.environ.copy()
    env['CUDA_VISIBLE_DEVICES'] = args.gpu

    subprocess_args = [
        'python3.8', os.path.join(cur_dir, 'agent.py'), 'train',
        '--config', os.path.join(args.dir, CONFIG_FILENAME),
        '--episodes', os.path.join(args.dir, EPISODES_DIR),
        '--wandb-id-file', os.path.join(args.dir, WANDB_ID_FILE),
    ]
    if not args.disable_checkpoints:
        subprocess_args.extend(['--checkpoints', os.path.join(args.dir, CHECKPOINTS_DIR)])
    if args.log_memory_transfer:
        subprocess_args.append('--log-memory-transfer')
    if args.log_profile:
        subprocess_args.append('--log-profile')
    if args.log_compilation:
        subprocess_args.append('--log-compilation')
    if args.disable_jit:
        subprocess_args.append('--disable-jit')
    if args.checkify_all:
        subprocess_args.append('--checkify-all')

    subprocess.run(env=env, args=subprocess_args)


def cleanup_experiment(args):
    if not os.path.exists(args.dir):
        raise RuntimeError(f'Directory {args.dir} not found')

    checkpoints_dir = os.path.join(args.dir, CHECKPOINTS_DIR)
    if os.path.exists(checkpoints_dir):
        shutil.rmtree(checkpoints_dir)
    else:
        logging.info(f'No checkpoints at {checkpoints_dir}')

    episodes_dir = os.path.join(args.dir, EPISODES_DIR)
    if os.path.exists(episodes_dir):
        shutil.rmtree(episodes_dir)
    else:
        logging.info(f'No episodes at {episodes_dir}')

    wandb_id_file = os.path.join(args.dir, WANDB_ID_FILE)
    if os.path.exists(wandb_id_file):
        os.remove(wandb_id_file)
    else:
        logging.info(f'No wandb id file at {wandb_id_file}')

    logging.info(f'Experiment at {args.dir} cleaned up!')


def play_episode(args):
    if not os.path.exists(args.file):
        raise RuntimeError(f'Episode file {args.file} not found')

    import nle
    nle_path = nle.__path__[0]
    subprocess.run(args=[
        'python3.8',
        os.path.join(nle_path, 'scripts', 'ttyplay.py'),
        '-s', f'{args.speed:.3f}', '-f', args.file,
    ])


def parse_args():
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(required=True, dest='mode', help='Available modes:')

    make_parser = subparsers.add_parser('make', help='Create a new experiment')
    make_parser.add_argument('--config', metavar='FILE', dest='config', required=True, type=str)
    make_parser.add_argument('--output-dir', metavar='DIR', dest='output_dir', required=True, type=str)
    make_parser.set_defaults(func=make_experiment)

    run_parser = subparsers.add_parser('run', help='Run an experiment')
    run_parser.add_argument('--dir', metavar='DIR', dest='dir', required=True, type=str)
    run_parser.add_argument('--gpu', metavar='GPU_NAME_OR_INDEX', dest='gpu', required=False, type=str, default='0')
    run_parser.add_argument('--log-memory-transfer', required=False, action='store_true', default=False)
    run_parser.add_argument('--log-profile', required=False, action='store_true', default=False)
    run_parser.add_argument('--log-compilation', required=False, action='store_true', default=False)
    run_parser.add_argument('--disable-jit', required=False, action='store_true', default=False)
    run_parser.add_argument('--disable-checkpoints', required=False, action='store_true', default=False)
    run_parser.add_argument('--checkify-all', action='store_true', required=False, default=False)
    run_parser.set_defaults(func=run_experiment)

    cleanup_parser = subparsers.add_parser('cleanup', help='Cleanup an experiment dir')
    cleanup_parser.add_argument('--dir', metavar='DIR', dest='dir', required=True, type=str)
    cleanup_parser.set_defaults(func=cleanup_experiment)

    play_parser = subparsers.add_parser('play', help='Play an episode')
    play_parser.add_argument('--file', metavar='FILE', dest='file', required=True, type=str)
    play_parser.add_argument('--speed', metavar='RATE', dest='speed', required=False, type=float, default=1.0)
    play_parser.set_defaults(func=play_episode)

    return parser.parse_args()


def main():
    args = parse_args()
    args.func(args)


if __name__ == '__main__':
    main()
