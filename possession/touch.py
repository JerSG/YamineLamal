"""Candidate feature: ball touches, from sudden changes in the ball's velocity.

A pass, a reception, a shot or a block changes the ball's velocity by several m/s in a fraction
of a second. With the velocity over a chord of TOUCH_CHORD_FRAMES before and after each frame,
a touch is a local peak of |v_after - v_before| above TOUCH_DV_MS, taken by a team when one of
its players is within TOUCH_REACH_M of the ball around the peak and clearly closer than any
opponent. Fixed physical rules: nothing is fitted to any match, so fit() learns nothing.

touch        momentary: the frames of a touch carry the touching team (or "contested");
             other frames with a ball are "none"; no ball is neutral
lasttouch    the team of the last touch, carried until the ball goes missing for over a second
             ("a loose ball stays with the team that last played it")
"""

import numpy as np

from . import config as C
from .features import Feature

TOUCH_CHORD_FRAMES = 12      # 0.48 s: long enough for a ball smoothed at a ~1 s scale
TOUCH_DV_MS = 3.0
TOUCH_REACH_M = 2.0
TOUCH_MARGIN_M = 0.75
TOUCH_WINDOW_FRAMES = 3
LOST_AFTER_FRAMES = 25


def touches(o):
    """Per frame: 0 home touch, 1 away touch, 2 contested, -1 none (peaks only)."""
    t = o["summary"]
    bx, by, seen = t["bx"], t["by"], t["ball_seen"]
    n, k = len(bx), TOUCH_CHORD_FRAMES
    out = np.full(n, -1, dtype=np.int8)
    if n <= 2 * k:
        return out
    ok = seen[:-2 * k] & seen[k:-k] & seen[2 * k:]
    vb = np.stack([bx[k:-k] - bx[:-2 * k], by[k:-k] - by[:-2 * k]], 1) * C.FPS / k
    va = np.stack([bx[2 * k:] - bx[k:-k], by[2 * k:] - by[k:-k]], 1) * C.FPS / k
    dv = np.full(n, np.nan)
    dv[k:-k] = np.where(ok, np.hypot(*(va - vb).T), np.nan)
    d0 = np.nan_to_num(dv, nan=-1.0)
    # local maximum over +/- k frames
    from numpy.lib.stride_tricks import sliding_window_view
    pad = np.concatenate([np.full(k, -1.0), d0, np.full(k, -1.0)])
    local_max = sliding_window_view(pad, 2 * k + 1).max(axis=1)
    peak = (d0 >= TOUCH_DV_MS) & (d0 == local_max)
    dh = np.where(np.isfinite(t["dh"]), t["dh"], np.inf)
    da = np.where(np.isfinite(t["da"]), t["da"], np.inf)
    w = TOUCH_WINDOW_FRAMES
    padh = np.concatenate([np.full(w, np.inf), dh, np.full(w, np.inf)])
    pada = np.concatenate([np.full(w, np.inf), da, np.full(w, np.inf)])
    mh = sliding_window_view(padh, 2 * w + 1).min(axis=1)
    ma = sliding_window_view(pada, 2 * w + 1).min(axis=1)
    home = (mh < TOUCH_REACH_M) & (mh + TOUCH_MARGIN_M < ma)
    away = (ma < TOUCH_REACH_M) & (ma + TOUCH_MARGIN_M < mh)
    contested = (mh < TOUCH_REACH_M) & (ma < TOUCH_REACH_M) & ~home & ~away
    out[peak & home] = 0
    out[peak & away] = 1
    out[peak & contested] = 2
    return out


def fit(train):
    return None


def touch_feature(o, state=None):
    tc = touches(o)
    seen = o["summary"]["ball_seen"]
    values = ["touch_home", "touch_away", "contested", "none", "unknown"]
    k = TOUCH_WINDOW_FRAMES
    spread = np.full(len(tc), -1, dtype=np.int8)
    for code in (0, 1, 2):          # mark +/- k frames around each touch
        idx = np.flatnonzero(tc == code)
        for d in range(-k, k + 1):
            j = np.clip(idx + d, 0, len(tc) - 1)
            spread[j] = code
    codes = np.select([spread == 0, spread == 1, spread == 2, seen], [0, 1, 2, 3], 4)
    init_team = {"touch_home": 0.06, "touch_away": 0.02, "contested": 0.02, "none": 0.90}
    init_dead = {"touch_home": 0.01, "touch_away": 0.01, "contested": 0.01, "none": 0.97}
    return [Feature("touch", values, codes.astype(np.int16), init_team, init_dead,
                    neutral=("unknown",))]


def lasttouch_feature(o, state=None):
    tc = touches(o)
    seen = o["summary"]["ball_seen"]
    n = len(tc)
    state_ = np.full(n, -1, dtype=np.int8)
    cur, lost = -1, 0
    for i in range(n):              # simple scan; ~150k frames per match
        if not seen[i]:
            lost += 1
            if lost > LOST_AFTER_FRAMES:
                cur = -1
        else:
            lost = 0
        if tc[i] in (0, 1):
            cur = tc[i]
        elif tc[i] == 2:
            cur = -1
        state_[i] = cur
    for s, e in o["blocks"]:        # nothing carried into a new block
        state_[s] = -1 if tc[s] < 0 else state_[s]
    values = ["last_home", "last_away", "unknown"]
    codes = np.select([state_ == 0, state_ == 1], [0, 1], 2)
    init_team = {"last_home": 0.75, "last_away": 0.25}
    init_dead = {"last_home": 0.50, "last_away": 0.50}
    return [Feature("lasttouch", values, codes.astype(np.int16), init_team, init_dead,
                    neutral=("unknown",))]
