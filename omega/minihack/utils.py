import numpy as np
import nle.nethack as nethack


BL_STAT_NAME_TO_INDEX = {
    'pos_x': nethack.NLE_BL_X,
    'pos_y': nethack.NLE_BL_Y,
    'strength_percentage': nethack.NLE_BL_STR25,
    'strength': nethack.NLE_BL_STR125,
    'dexterity': nethack.NLE_BL_DEX,
    'constitution': nethack.NLE_BL_CON,
    'intelligence': nethack.NLE_BL_INT,
    'wisdom': nethack.NLE_BL_WIS,
    'charisma': nethack.NLE_BL_CHA,
    'score': nethack.NLE_BL_SCORE,
    'hitpoints': nethack.NLE_BL_HP,
    'max_hitpoints': nethack.NLE_BL_HPMAX,
    'depth': nethack.NLE_BL_DEPTH,
    'gold': nethack.NLE_BL_GOLD,
    'energy': nethack.NLE_BL_ENE,
    'max_energy': nethack.NLE_BL_ENEMAX,
    'armor_class': nethack.NLE_BL_AC,
    'monster_level': nethack.NLE_BL_HD,
    'experience_level': nethack.NLE_BL_XP,
    'experience_points': nethack.NLE_BL_EXP,
    'time': nethack.NLE_BL_TIME,
    'hunger_state': nethack.NLE_BL_HUNGER,
    'carrying_capacity': nethack.NLE_BL_CAP,
    'dungeon_number': nethack.NLE_BL_DNUM,
    'level_number': nethack.NLE_BL_DLEVEL,
    'condition': nethack.NLE_BL_CONDITION,
}


def filtered_bl_stats_shape(keys_to_filter=None, keys_to_keep=None):
    if keys_to_filter is not None:
        return len(BL_STAT_NAME_TO_INDEX) - len(keys_to_filter),
    if keys_to_keep is not None:
        return len(keys_to_keep),

    raise ValueError('Either keys_to_filter or keys_to_keep must be specified, but not both')


def keep_bl_stats(bl_stats, keys_to_keep):
    assert all(k in BL_STAT_NAME_TO_INDEX for k in keys_to_keep)
    assert bl_stats.shape[-1] == len(BL_STAT_NAME_TO_INDEX)
    indices_to_keep = [index for name, index in BL_STAT_NAME_TO_INDEX.items() if name in keys_to_keep]
    return bl_stats[..., indices_to_keep]


def filter_bl_stats(bl_stats, keys_to_filter):
    assert all(k in BL_STAT_NAME_TO_INDEX for k in keys_to_filter)
    keys_to_keep = [k for k in BL_STAT_NAME_TO_INDEX.keys() if k not in keys_to_filter]
    return keep_bl_stats(bl_stats, keys_to_keep)


def glyph_pos(glyphs, glyph):
    glyph_positions = np.where(np.asarray(glyphs) == glyph)
    assert len(glyph_positions) == 2
    if glyph_positions[0].shape[0] == 0:
        return None
    return np.array([glyph_positions[0][0], glyph_positions[1][0]], dtype=np.float32)


def print_char_glyphs(char_glyphs):
    for r in range(char_glyphs.shape[0]):
        for c in range(char_glyphs.shape[1]):
            print(chr(char_glyphs[r][c]), end='')
        print('\n')
