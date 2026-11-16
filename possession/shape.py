"""Candidate feature: which team has the ball, from the shape and movement of the two teams.

Where the ball is not seen, the teams still show who has it: the side in possession pushes up,
spreads wide and moves forward; the other side drops and narrows. For every frame, five
home-minus-away differences are computed from the players only, each oriented by the direction
of attack (inferred per block from where each team stands), over five time windows:

  quantities   forward position of the centroid, spread in x, spread in y, mean speed,
               forward velocity of the centroid
  windows      centred means over 1 s and 4 s, and 2 s means shifted 1 s and 2 s into the
               future and 1 s into the past (a team reacts to a change of possession late)

A logistic regression (L2, no intercept) maps the 25 columns to P(home has the ball) on live
frames. Every column changes sign when home and away are exchanged and there is no intercept,
so the model is exactly antisymmetric: it cannot prefer the home side of the matches it was
trained on. It is fitted on the external training games only, and applied unchanged. The
logit is binned into five levels with fixed edges and used as one more categorical feature.
"""

import warnings

import numpy as np

from . import config as C
from .features import Feature, centred_mean
from .sources import HOME, AWAY

WINDOWS = ((25, 0), (100, 0), (50, 25), (50, 50), (50, -25))   # (width frames, shift frames)
L2 = 1.0
SUBSAMPLE = 5
LOGIT_EDGES = (-2.0, -0.5, 0.5, 2.0)
MIN_PLAYERS = 5


def _attacks_right(px, slot_team, blocks, n):
    right = np.zeros(n, dtype=bool)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        for s, e in blocks:
            h = np.nanmean(px[s:e][:, slot_team == HOME])
            a = np.nanmean(px[s:e][:, slot_team == AWAY])
            right[s:e] = bool(h < a)
    return right


def raw_quantities(o, pitch):
    g = o["game"]
    length, width = pitch
    px = g["players"][..., 0].astype(np.float64) * length
    py = g["players"][..., 1].astype(np.float64) * width
    seen = np.isfinite(px) & np.isfinite(py)
    n = len(px)
    st = np.asarray(g["slot_team"])
    right = _attacks_right(px, st, o["blocks"], n)
    from .strict import speed_from_positions
    sp = speed_from_positions(g["players"], pitch).astype(np.float64)

    def team(mask, forward_sign):
        k = seen[:, mask].sum(axis=1)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            cx = np.nanmean(np.where(seen[:, mask], px[:, mask], np.nan), axis=1)
            sx = np.nanstd(np.where(seen[:, mask], px[:, mask], np.nan), axis=1)
            sy = np.nanstd(np.where(seen[:, mask], py[:, mask], np.nan), axis=1)
            v = np.nanmean(np.where(seen[:, mask], sp[:, mask], np.nan), axis=1)
        fwd = np.where(forward_sign > 0, cx, length - cx)
        vel = np.full(n, np.nan)
        vel[1:] = np.diff(fwd) * C.FPS
        bad = k < MIN_PLAYERS
        for arr in (fwd, sx, sy, v, vel):
            arr[bad] = np.nan
        return fwd, sx, sy, v, vel

    sign_h = np.where(right, 1.0, -1.0)
    th = team(st == HOME, sign_h)
    ta = team(st == AWAY, -sign_h)
    diffs = [h - a for h, a in zip(th, ta)]
    for s, e in o["blocks"]:         # no velocity across block edges
        diffs[4][s] = np.nan
    return diffs


def _window(x, width, shift, blocks):
    out = np.full(len(x), np.nan)
    for s, e in blocks:
        m = centred_mean(x[s:e], width // 2)
        if shift > 0:
            m = np.concatenate([m[shift:], np.full(min(shift, len(m)), np.nan)])
        elif shift < 0:
            m = np.concatenate([np.full(min(-shift, len(m)), np.nan), m[:shift]])
        out[s:e] = m
    return out


def design(o, pitch=None):
    pitch = pitch or o.get("pitch") or C.STRICT_EXTERNAL_PITCH
    if "_shape_X" in o:
        return o["_shape_X"]
    diffs = raw_quantities(o, pitch)
    cols = [_window(d, w, sh, o["blocks"]) for d in diffs for w, sh in WINDOWS]
    X = np.stack(cols, axis=1)
    o["_shape_X"] = X
    return X


def _logistic(X, y, l2):
    w = np.zeros(X.shape[1])
    for _ in range(25):                     # Newton steps
        z = np.clip(X @ w, -30, 30)
        p = 1.0 / (1.0 + np.exp(-z))
        g = X.T @ (p - y) + l2 * w
        Hm = (X * (p * (1 - p))[:, None]).T @ X + l2 * np.eye(len(w))
        step = np.linalg.solve(Hm, g)
        w -= step
        if np.abs(step).max() < 1e-6:
            break
    return w


def fit(train):
    Xs, ys = [], []
    for o in train:
        X = design(o)
        lab = o["labels"]
        ok = np.isfinite(X).all(axis=1) & ((lab == 0) | (lab == 1))
        idx = np.flatnonzero(ok)[::SUBSAMPLE]
        Xs.append(X[idx])
        ys.append((lab[idx] == 0).astype(float))
    X = np.concatenate(Xs)
    y = np.concatenate(ys)
    scale = np.sqrt((X ** 2).mean(axis=0)) + 1e-9      # no centring: keeps antisymmetry
    w = _logistic(X / scale, y, L2 * len(y) / 1e4)
    acc = float(np.mean(((X / scale) @ w > 0) == (y > 0.5)))
    return {"w": w, "scale": scale, "train_accuracy": acc}


def logit(o, state):
    X = design(o)
    z = (X / state["scale"]) @ state["w"]
    z[~np.isfinite(X).all(axis=1)] = np.nan
    return z


def shape_feature(o, state):
    z = logit(o, state)
    values = ["home_strong", "home", "even", "away", "away_strong", "unknown"]
    codes = np.searchsorted(np.asarray(LOGIT_EDGES), np.nan_to_num(z), side="right")
    codes = 4 - codes                                  # high logit = home
    codes = np.where(np.isfinite(z), codes, 5).astype(np.int16)
    init_team = {"home_strong": 0.30, "home": 0.25, "even": 0.20, "away": 0.15,
                 "away_strong": 0.10}
    init_dead = {"home_strong": 0.15, "home": 0.20, "even": 0.30, "away": 0.20,
                 "away_strong": 0.15}
    return [Feature("shape", values, codes, init_team, init_dead, neutral=("unknown",))]


# ------------------------------------------------------------------------------------------------
# a second model for frames without a ball
# ------------------------------------------------------------------------------------------------
# Where the ball is not seen, the team shape is not the only clue: where the ball was last seen
# and which team last touched it still say a lot, and less the longer ago that was. The no-ball
# model adds four columns to the 25 shape columns, each changing sign when the teams are
# exchanged: the last-seen ball position along the home team's attacking direction (-0.5 own
# goal .. +0.5 opponent's goal), the last touching team (+1 home, -1 away, 0 contested or none),
# and both multiplied by exp(-seconds since the ball was last seen / NOBALL_DECAY_S). It is
# fitted on live frames without a ball only, and applies only there (neutral elsewhere).
NOBALL_DECAY_S = 5.0
NOBALL_SUBSAMPLE = 2


def noball_columns(o):
    if "_noball_X" in o:
        return o["_noball_X"]
    from . import touch
    pitch = o.get("pitch") or C.STRICT_EXTERNAL_PITCH
    g, t = o["game"], o["summary"]
    n = len(t["ball_seen"])
    px = g["players"][..., 0].astype(np.float64) * pitch[0]
    right = _attacks_right(px, np.asarray(g["slot_team"]), o["blocks"], n)
    seen = t["ball_seen"]
    bx = t["bx"] / pitch[0]
    pos = np.where(right, bx - 0.5, 0.5 - bx)                   # home attacking direction
    tc = touch.touches(o)
    sign = np.where(tc == 0, 1.0, np.where(tc == 1, -1.0, np.where(tc == 2, 0.0, np.nan)))
    lbx = np.full(n, np.nan)
    lt = np.full(n, np.nan)
    age = np.full(n, np.nan)
    for s, e in o["blocks"]:
        last_pos, last_t, last_seen = np.nan, 0.0, None
        for i in range(s, e):                                    # one pass per block
            if seen[i]:
                last_pos, last_seen = pos[i], i
            if np.isfinite(sign[i]):
                last_t = sign[i]
            lbx[i] = last_pos
            lt[i] = last_t
            age[i] = (i - last_seen) / C.FPS if last_seen is not None else np.nan
    w = np.exp(-age / NOBALL_DECAY_S)
    extra = np.stack([lbx, lt, lbx * w, lt * w], axis=1)
    X = np.concatenate([design(o), extra], axis=1)
    o["_noball_X"] = X
    return X


def fit_noball(train):
    Xs, ys = [], []
    for o in train:
        X = noball_columns(o)
        lab = o["labels"]
        ok = np.isfinite(X).all(axis=1) & ((lab == 0) | (lab == 1)) & ~o["summary"]["ball_seen"]
        idx = np.flatnonzero(ok)[::NOBALL_SUBSAMPLE]
        Xs.append(X[idx])
        ys.append((lab[idx] == 0).astype(float))
    X = np.concatenate(Xs)
    y = np.concatenate(ys)
    scale = np.sqrt((X ** 2).mean(axis=0)) + 1e-9
    w = _logistic(X / scale, y, L2 * len(y) / 1e4)
    acc = float(np.mean(((X / scale) @ w > 0) == (y > 0.5)))
    return {"w": w, "scale": scale, "train_accuracy": acc}


def noball_feature(o, state):
    X = noball_columns(o)
    z = (X / state["scale"]) @ state["w"]
    z[~np.isfinite(X).all(axis=1) | o["summary"]["ball_seen"]] = np.nan
    values = ["home_strong", "home", "even", "away", "away_strong", "unknown"]
    codes = 4 - np.searchsorted(np.asarray(LOGIT_EDGES), np.nan_to_num(z), side="right")
    codes = np.where(np.isfinite(z), codes, 5).astype(np.int16)
    init_team = {"home_strong": 0.30, "home": 0.25, "even": 0.20, "away": 0.15,
                 "away_strong": 0.10}
    init_dead = {"home_strong": 0.15, "home": 0.20, "even": 0.30, "away": 0.20,
                 "away_strong": 0.15}
    return [Feature("noball", values, codes, init_team, init_dead, neutral=("unknown",))]


def shape_seen_feature(o, state):
    """The shape feature on frames with a ball only (for use together with noball_feature)."""
    f = shape_feature(o, state)[0]
    codes = np.where(o["summary"]["ball_seen"], f.codes, len(f.values) - 1).astype(np.int16)
    return [Feature("shape", f.values, codes, dict(zip(f.values, f.init_team)),
                    dict(zip(f.values, f.init_dead)), neutral=("unknown",))]
