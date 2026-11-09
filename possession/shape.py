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
