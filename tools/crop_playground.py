import gym
import minihack
import nle.nethack

from omega.training.env_wrapper import EnvWrapper

import numpy as np


ENV_NAME = 'MiniHack-Memento-Short-F2-v0'
NUM_STEPS = 50
GLYPH_CROP_START = [0, 0]
GLYPH_CROP_AREA = [13, 20]


def generate_random_action(env):
    return np.random.randint(low=0, high=env.action_space.n)


def crop_observation(observation):
    if GLYPH_CROP_START is None:
        # Can be used to crop unused observation area to speedup convergence
        start_r = (nle.nethack.DUNGEON_SHAPE[0] - GLYPH_CROP_AREA[0]) // 2
        start_c = (nle.nethack.DUNGEON_SHAPE[1] - GLYPH_CROP_AREA[1]) // 2
    else:
        start_r, start_c = GLYPH_CROP_START
    return observation[start_r:start_r + GLYPH_CROP_AREA[0], start_c:start_c + GLYPH_CROP_AREA[1]]


def print_observation(observation):
    assert len(observation.shape) == 2
    for c in range(observation.shape[1] + 2):
        print('X', end='')
    print('\n')
    for r in range(observation.shape[0]):
        print('X', end='')
        for c in range(observation.shape[1]):
            print(chr(observation[r][c]), end='')
        print('X\n')
    for c in range(observation.shape[1] + 2):
        print('X', end='')
    print('\n')


def print_cropped_observation(env):
    observation = env.current_state['chars']
    cropped_observation = crop_observation(observation)
    print_observation(cropped_observation)


def main():
    env = gym.make(ENV_NAME)
    env = EnvWrapper(env)
    env.reset()
    for step in range(NUM_STEPS):
        print_cropped_observation(env)
        _, _, done, _ = env.step(generate_random_action(env))
        if done:
            env.reset()


if __name__ == '__main__':
    main()
