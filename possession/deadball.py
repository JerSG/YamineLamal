"""Dead-ball features: a still ball at a restart place and the restart kick (used), and gap
context (tested, not used).

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


KICK_SPEED_MS = 5.0         # the ball leaves faster than this when a restart is taken
KICK_SEARCH_FRAMES = (-12, 25)   # where to look for the kick around the end of a still run
KICK_MARK_FRAMES = 25       # frames marked after the kick (one second)
KICK_REACH_M = 2.0
KICK_MARGIN_M = 0.75


def restart_feature(t):
    """The second after a still ball (still for STILL_MIN_S) is struck: play has restarted, and
    the team of the player at the ball when it was struck usually has it. Values: kick by home,
    kick by away, kick with no clear taker, none (ball seen), unknown (no ball: neutral)."""
    v = t["ball_speed"]
    seen = t["ball_seen"]
    n = len(v)
    with np.errstate(invalid="ignore"):
        slow = seen & np.isfinite(v) & (v < STILL_SPEED_MS)
        fast = seen & np.isfinite(v) & (v > KICK_SPEED_MS)
    still = slow & (run_lengths(slow) >= STILL_MIN_S * C.FPS)
    ends = np.flatnonzero(still[:-1] & ~still[1:]) + 1          # first frame after a still run
    dh = np.where(np.isfinite(t["dh"]), t["dh"], np.inf)
    da = np.where(np.isfinite(t["da"]), t["da"], np.inf)
    mark = np.full(n, -1, dtype=np.int8)
    lo_off, hi_off = KICK_SEARCH_FRAMES
    for e in ends:
        lo, hi = max(0, e + lo_off), min(n, e + hi_off)
        k = np.flatnonzero(fast[lo:hi])
        if not len(k):
            continue
        kick = lo + k[0]
        a, b = max(0, e - 12), min(n, e + 12)
        mh, ma = dh[a:b].min(), da[a:b].min()
        team = 0 if (mh < KICK_REACH_M and mh + KICK_MARGIN_M < ma) else \
            1 if (ma < KICK_REACH_M and ma + KICK_MARGIN_M < mh) else 2
        mark[kick:min(n, kick + KICK_MARK_FRAMES)] = team
    values = ["kick_home", "kick_away", "kick_unclear", "none", "unknown"]
    codes = np.select([mark == 0, mark == 1, mark == 2, seen], [0, 1, 2, 3], 4)
    init_team = {"kick_home": 0.03, "kick_away": 0.01, "kick_unclear": 0.01, "none": 0.95}
    init_dead = {"kick_home": 0.005, "kick_away": 0.005, "kick_unclear": 0.005, "none": 0.985}
    return Feature("restart", values, codes.astype(np.int16), init_team, init_dead,
                   neutral=("unknown",))
