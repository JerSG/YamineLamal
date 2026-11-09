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
PROFILES = {"markov": None, "short": 0.5, "long": 4.0, "crowded": 2.0, "smooth": None}
SMOOTH_FRAMES = 25


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
    g["ball"] = ball
    return g
