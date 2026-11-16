"""Algorithmic correction of the ball track, applied before the features (no learning).

The competition's tracking is automatic and uncorrected. Two fixed, physical rules repair the
ball track; both use the tracking alone, with thresholds set in advance from football facts:

  glitches   a ball position more than GLITCH_M away from both of its neighbours' midpoint, when
             both neighbours are tracked, is a false detection and is removed
  short gaps a ball missing for at most GAP_MAX_FRAMES, tracked on both sides, is filled with the
             straight line between the two, when that flight is physically possible (under
             MAX_BALL_SPEED_MS)

Nothing here reads labels, ground truth or video, and nothing is fitted to any match.
"""

import numpy as np

from . import config as C

GLITCH_M = 5.0           # a real ball cannot jump this far from its neighbours in 1/25 s and back
GAP_MAX_FRAMES = 25      # fill gaps of up to one second
MAX_BALL_SPEED_MS = 45.0


def correct_ball(ball, pitch, blocks=None):
    """A corrected copy of the ball array (T, >=2, normalised coordinates)."""
    length, width = pitch
    out = ball.copy()
    x = out[:, 0].astype(np.float64) * length
    y = out[:, 1].astype(np.float64) * width
    seen = np.isfinite(x) & np.isfinite(y)
    # 1. single-frame glitches
    mid_x = (x[:-2] + x[2:]) / 2
    mid_y = (y[:-2] + y[2:]) / 2
    with np.errstate(invalid="ignore"):
        far = np.hypot(x[1:-1] - mid_x, y[1:-1] - mid_y) > GLITCH_M
    glitch = np.zeros(len(x), dtype=bool)
    glitch[1:-1] = far & seen[:-2] & seen[1:-1] & seen[2:]
    out[glitch, :2] = np.nan
    seen &= ~glitch
    # 2. short gaps, inside each block only
    spans = blocks or [(0, len(out))]
    idx = np.flatnonzero(seen)
    for a, b in zip(idx[:-1], idx[1:]):
        gap = b - a - 1
        if gap < 1 or gap > GAP_MAX_FRAMES:
            continue
        if not any(s <= a and b < e for s, e in spans):
            continue
        xa, ya = out[a, 0] * length, out[a, 1] * width
        xb, yb = out[b, 0] * length, out[b, 1] * width
        if np.hypot(xb - xa, yb - ya) * C.FPS / (b - a) > MAX_BALL_SPEED_MS:
            continue
        f = (np.arange(1, gap + 1) / (b - a))[:, None]
        out[a + 1:b, :2] = out[a, :2] + f * (out[b, :2] - out[a, :2])
    return out
