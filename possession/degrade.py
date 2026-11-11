"""Tracking-gap profiles for testing on external matches whose ball is (almost) always tracked.

The DFL open data tracks the ball on nearly every frame; the competition's automatic tracking
loses it on 41% of graded frames (the organisers' DATA.md). To test a model on DFL under
realistic and harsher conditions, ball positions are removed according to a profile:

  markov    a two-state chain whose probabilities are measured on the Metrica external games,
            separately for live and dead play (possession/dfl.py); dead play loses the ball far
            more often, as automatic tracking does
  short     gaps of mean 0.5 s, placed anywhere, until 41% of each block is missing
  long      gaps of mean 4 s, the same share
  crowded   gaps of mean 2 s, more likely to start where many players are near the ball
  smooth    the markov gaps, and the remaining ball positions smoothed over about one second
            (a centred mean of SMOOTH_FRAMES within each tracked stretch), because the
            competition's automatic tracking delivers a smoothed ball
  full      the crowded gaps, plus what automatic tracking does to players: every player's
            track is cut into fragments (mean FRAGMENT_MEAN_S) that land in anonymous slots,
            SLOTS_PER_TEAM per team as in the competition's 46, and Gaussian position noise
            (PLAYER_NOISE_M for players, BALL_NOISE_M for the ball)

Every profile is deterministic for a (match, profile, seed). Labels are never read by the gap
placement except by `markov`, whose live/dead rates come from the external games; the others
see positions only. Nothing here reads competition data.
"""

import zlib

import numpy as np

from . import config as C

MISSING_SHARE = 0.41        # organisers' DATA.md: no ball on 41% of graded frames
MAX_GAP_S = 30.0
CROWD_RADIUS_M = 5.0
PROFILES = {"markov": None, "short": 0.5, "long": 4.0, "crowded": 2.0, "smooth": None,
            "full": 2.0}
SMOOTH_FRAMES = 25
SLOTS_PER_TEAM = 23
FRAGMENT_MEAN_S = 60.0
PLAYER_NOISE_M = 0.10
BALL_NOISE_M = 0.20


def rng_for(name, profile, seed):
    return np.random.RandomState(
        (zlib.crc32("{}|{}|{}".format(name, profile, seed).encode()) & 0x7FFFFFFF))


def crowd_weights(players, ball, pitch):
    length, width = pitch
    dx = (players[..., 0] - ball[:, None, 0]) * length
    dy = (players[..., 1] - ball[:, None, 1]) * width
    with np.errstate(invalid="ignore"):
        near = np.hypot(dx, dy) < CROWD_RADIUS_M
    return near.sum(axis=1).astype(np.float64)


def gap_mask(blocks, n, mean_s, rng, weights=None, share=MISSING_SHARE):
    """Missing-ball mask: exponential gap lengths of mean mean_s, starts uniform (or weighted),
    added until each block has `share` of its frames missing."""
    missing = np.zeros(n, dtype=bool)
    cap = int(MAX_GAP_S * C.FPS)
    for s, e in blocks:
        target = int(round(share * (e - s)))
        count = 0
        while count < target:
            free = np.flatnonzero(~missing[s:e]) + s
            if weights is not None and weights[free].sum() > 0:
                p = weights[free] + 1e-3
                start = int(rng.choice(free, p=p / p.sum()))
            else:
                start = int(free[rng.randint(len(free))])
            length = max(1, min(int(np.ceil(rng.exponential(mean_s * C.FPS))), cap,
                                target - count, e - start))
            seg = missing[start:start + length]
            count += int((~seg).sum())
            seg[:] = True
    return missing


def smooth_ball(ball, frames=SMOOTH_FRAMES):
    """Centred mean of x and y over `frames`, within each run of tracked frames."""
    from .features import centred_mean
    out = ball.copy()
    seen = np.isfinite(ball[:, 0])
    edges = np.flatnonzero(np.diff(np.concatenate(([0], seen.astype(int), [0]))))
    for a, b in zip(edges[::2], edges[1::2]):
        for c in (0, 1):
            out[a:b, c] = centred_mean(ball[a:b, c].astype(np.float64), frames // 2)
    return out


def fragment_players(players, slot_team, rng, pitch):
    """Anonymous fragmented tracks: (T, 2 * SLOTS_PER_TEAM, 3) and the new slot -> team map.
    Each player's tracked stretches are cut into exponential fragments; each fragment goes to a
    slot of the player's team that is free for its whole duration."""
    T = players.shape[0]
    out = np.full((T, 2 * SLOTS_PER_TEAM, 3), np.nan, dtype=np.float32)
    new_team = np.array([0] * SLOTS_PER_TEAM + [1] * SLOTS_PER_TEAM, dtype=np.int8)
    free_from = np.zeros(2 * SLOTS_PER_TEAM, dtype=np.int64)
    pieces = []
    for p in range(players.shape[1]):
        seen = np.isfinite(players[:, p, 0])
        edges = np.flatnonzero(np.diff(np.concatenate(([0], seen.astype(int), [0]))))
        for a, b in zip(edges[::2], edges[1::2]):
            t = a
            while t < b:
                n = max(25, int(rng.exponential(FRAGMENT_MEAN_S * C.FPS)))
                pieces.append((t, min(b, t + n), p))
                t += n
    pieces.sort()
    for a, b, p in pieces:
        team = int(slot_team[p])
        slots = np.flatnonzero((new_team == team) & (free_from <= a))
        if not len(slots):
            continue                       # more simultaneous fragments than slots: dropped
        k = int(slots[rng.randint(len(slots))])
        out[a:b, k] = players[a:b, p]
        free_from[k] = b
    length, width = pitch
    noise = rng.normal(0, 1, out.shape[:2] + (2,)).astype(np.float32)
    out[..., 0] += noise[..., 0] * PLAYER_NOISE_M / length
    out[..., 1] += noise[..., 1] * PLAYER_NOISE_M / width
    return out, new_team


def apply_profile(game, profile, seed, markov_model=None, pitch=(105.0, 68.0)):
    """A copy of the game with the ball removed according to the profile."""
    from . import dfl
    from .strict import blocks_from_labels
    tag = "{}|{}".format(profile, seed)
    if profile == "markov":
        return dfl.apply_missing(game, markov_model, game["name"] + "|" + tag)
    if profile == "smooth":
        g = dfl.apply_missing(game, markov_model, game["name"] + "|" + tag)
        g["ball"] = smooth_ball(g["ball"])
        return g
    rng = rng_for(game["name"], profile, seed)
    blocks = blocks_from_labels(game["labels"])
    w = crowd_weights(game["players"], game["ball"], pitch) if profile == "crowded" else None
    missing = gap_mask(blocks, len(game["ball"]), PROFILES[profile], rng, w)
    g = dict(game)
    ball = game["ball"].copy()
    ball[missing] = np.nan
    if profile == "full":
        length, width = pitch
        ball[:, 0] += rng.normal(0, BALL_NOISE_M / length, len(ball)).astype(np.float32)
        ball[:, 1] += rng.normal(0, BALL_NOISE_M / width, len(ball)).astype(np.float32)
        g["players"], g["slot_team"] = fragment_players(game["players"], game["slot_team"],
                                                         rng, pitch)
    g["ball"] = ball
    return g
