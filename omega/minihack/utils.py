import numpy as np


def glyph_pos(glyphs, glyph):
    glyph_positions = np.where(np.asarray(glyphs) == glyph)
    assert len(glyph_positions) == 2
    if glyph_positions[0].shape[0] == 0:
        return None
    return np.array([glyph_positions[0][0], glyph_positions[1][0]], dtype=np.float)
