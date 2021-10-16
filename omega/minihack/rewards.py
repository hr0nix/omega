import numpy

from .utils import glyph_pos


def distance_to_staircase_reward(env, prev_obs, action, current_obs):
    glyphs = current_obs[env._observation_keys.index('chars')]
    cur_pos = glyph_pos(glyphs, ord('@'))
    staircase_pos = glyph_pos(glyphs, ord('>'))
    if staircase_pos is None:
        # Staircase has been reached
        return 0.0
    distance = numpy.linalg.norm(cur_pos - staircase_pos)
    distance /= numpy.max(glyphs.shape)
    return -distance
