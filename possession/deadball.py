"""Dead-ball features: a still ball at a restart place (used), and gap context (tested, not used).

Both are fixed geometric rules (pitch markings in metres, no thresholds fitted to any match).
Their emission tables are learned on the external games like every other tracking feature.
`still` is part of the submitted model; `gapctx` added nothing measurable on the benchmark once
`still` was in (scripts/benchmark.py, docs/TECHNICAL_REPORT.md 4.9) and is kept for reference.

gapctx   for every frame of a ball gap (a run of frames with no ball inside graded play): was the
         ball last seen at or over a line before the gap (exit), and is it first seen again at a
         restart place after it (entry: at or over a line, at a spot, or in a goal area)?
         -> both / exit_only / entry_only / neither; frames with a ball are neutral ("seen").
         Offline, so looking at where the gap ends is allowed (the decoder sees the whole match).
still    for frames with a ball: is it stationary (ball speed below STILL_SPEED_MS for at least
         STILL_MIN_S) and where: at a restart place, or in open play; or moving. A missing ball
         is neutral.
"""

import numpy as np

from . import config as C
from .features import Feature, run_lengths

LINE_M = 2.0            # within this distance of a touchline or goal line, or beyond it
SPOT_M = 2.5            # within this distance of the centre spot or a penalty spot
PENALTY_SPOT_M = 11.0
GOAL_AREA_DEPTH_M = 5.5
GOAL_AREA_HALF_WIDTH_M = 9.16
STILL_SPEED_MS = 1.0
STILL_MIN_S = 1.0
NAMES = ("gapctx", "still")


def restart_place(x, y, pitch):
    """True where a restart is taken: at or beyond a line, at a spot, or in a goal area."""
    length, width = pitch
    with np.errstate(invalid="ignore"):
        to_line = np.minimum.reduce([x, length - x, y, width - y])
        line = to_line < LINE_M
        spots = [(length / 2, width / 2), (PENALTY_SPOT_M, width / 2),
                 (length - PENALTY_SPOT_M, width / 2)]
        spot = np.zeros(np.shape(x), dtype=bool)
        for sx, sy in spots:
            spot |= np.hypot(x - sx, y - sy) < SPOT_M
        goal_area = ((x < GOAL_AREA_DEPTH_M) | (x > length - GOAL_AREA_DEPTH_M)) & \
            (np.abs(y - width / 2) < GOAL_AREA_HALF_WIDTH_M)
    return np.isfinite(x) & np.isfinite(y) & (line | spot | goal_area)


def _carry(valid, backward=False):
    """Index of the last (or, backward, the next) valid frame for every frame; -1 if none."""
    n = len(valid)
    if backward:
        idx = _carry(valid[::-1])[::-1]
        return np.where(idx >= 0, n - 1 - idx, -1)
    idx = np.where(valid, np.arange(n), -1)
    np.maximum.accumulate(idx, out=idx)
    return idx


def gapctx_feature(t, blocks):
    seen = t["ball_seen"]
    place = restart_place(t["bx"], t["by"], t["pitch"])
    values = ["both", "exit_only", "entry_only", "neither", "seen"]
    codes = np.full(len(seen), 4, dtype=np.int16)
    for s, e in blocks:
        sv = seen[s:e]
        before = _carry(sv)
        after = _carry(sv, backward=True)
        exit_ok = np.where(before >= 0, place[s:e][np.maximum(before, 0)], False)
        entry_ok = np.where(after >= 0, place[s:e][np.maximum(after, 0)], False)
        c = np.select([exit_ok & entry_ok, exit_ok, entry_ok], [0, 1, 2], 3)
        codes[s:e] = np.where(sv, 4, c)
    init_team = {"both": 0.05, "exit_only": 0.15, "entry_only": 0.15, "neither": 0.65}
    init_dead = {"both": 0.40, "exit_only": 0.20, "entry_only": 0.25, "neither": 0.15}
    return Feature("gapctx", values, codes, init_team, init_dead, team_bearing=False,
                   neutral=("seen",))


def still_feature(t):
    v = t["ball_speed"]
    seen = t["ball_seen"]
    with np.errstate(invalid="ignore"):
        slow = seen & np.isfinite(v) & (v < STILL_SPEED_MS)
    still = slow & (run_lengths(slow) >= STILL_MIN_S * C.FPS)
    place = restart_place(t["bx"], t["by"], t["pitch"])
    values = ["still_restart", "still_open", "moving", "unknown"]
    codes = np.select([still & place, still, seen], [0, 1, 2], 3).astype(np.int16)
    init_team = {"still_restart": 0.01, "still_open": 0.04, "moving": 0.95}
    init_dead = {"still_restart": 0.25, "still_open": 0.15, "moving": 0.60}
    return Feature("still", values, codes, init_team, init_dead, team_bearing=False,
                   neutral=("unknown",))


def features(t, blocks, names=NAMES):
    out = []
    if "gapctx" in names:
        out.append(gapctx_feature(t, blocks))
    if "still" in names:
        out.append(still_feature(t))
    return out
