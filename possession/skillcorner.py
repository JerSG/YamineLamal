"""SkillCorner open data: broadcast tracking of 20 A-League matches, as an external test set.

    SkillCorner open data, https://github.com/SkillCorner/opendata (MIT licence; the data is
    credited to SkillCorner as its README asks). Tracking at 10 fps from the broadcast video.

Why it is useful: broadcast tracking has *real* tracker failures. Players off screen are not
detected (about 12 of 22 on a typical frame) and the ball is detected on about 60% of frames, so
it tests the model on genuine gaps rather than simulated ones. Only detected objects are kept:
SkillCorner's own extrapolated positions are dropped, so a gap stays a gap.

Labels: every frame names the team in possession ("home team" / "away team") or none. "None"
covers both dead ball and loose balls, so it cannot be used as `D`; those frames are left
ungraded (X). The set therefore tests who has the ball, not dead ball. It is used for testing
only, never for training or tuning.

The 10 fps frames are interpolated linearly to the 25 fps the model works at, only between two
detected frames of the same object; labels take the nearest 10 fps frame.
"""

import json
import os

import numpy as np

H, A, D, X = 0, 1, 2, 3
SRC_FPS, FPS = 10.0, 25.0


def _upsample(v, n_out):
    """v: (T10, ...) with NaN gaps -> (n_out, ...) linear between two finite samples only."""
    t = np.arange(n_out) * SRC_FPS / FPS
    i0 = np.minimum(np.floor(t).astype(int), len(v) - 1)
    i1 = np.minimum(i0 + 1, len(v) - 1)
    f = (t - i0).reshape((-1,) + (1,) * (v.ndim - 1))
    a, b = v[i0], v[i1]
    out = a * (1 - f) + b * f
    exact = (f == 0)
    out = np.where(exact, a, out)
    bad = ~np.isfinite(a) | (~np.isfinite(b) & ~exact)
    out[bad] = np.nan
    return out.astype(np.float32)


def load_game(root, tracking_dir, match_id):
    """One match in the common format of possession/external.py, with labels (H/A/X)."""
    mid = str(match_id)
    meta = json.load(open(os.path.join(root, "data", "matches", mid, mid + "_match.json")))
    length, width = float(meta["pitch_length"]), float(meta["pitch_width"])
    home_id = meta["home_team"]["id"]
    players = sorted(meta["players"], key=lambda p: (p["team_id"] != home_id, p["id"]))
    slot = {p["id"]: i for i, p in enumerate(players)}
    slot_team = np.array([H if p["team_id"] == home_id else A for p in players], dtype=np.int8)
    frames = {1: [], 2: []}
    with open(os.path.join(tracking_dir, mid + "_tracking.jsonl")) as handle:
        for line in handle:
            fr = json.loads(line)
            if fr["period"] in frames:
                frames[fr["period"]].append(fr)
    chunks_p, chunks_b, chunks_l, period = [], [], [], []
    for per in (1, 2):
        fs = frames[per]
        if not fs:
            continue
        n10 = len(fs)
        P = np.full((n10, len(players), 2), np.nan, dtype=np.float64)
        B = np.full((n10, 2), np.nan, dtype=np.float64)
        lab10 = np.full(n10, X, dtype=np.int8)
        for i, fr in enumerate(fs):
            for p in fr["player_data"]:
                k = slot.get(p["player_id"])
                if k is not None and p.get("is_detected"):
                    P[i, k] = ((p["x"] + length / 2) / length, (p["y"] + width / 2) / width)
            bd = fr["ball_data"]
            if bd.get("is_detected") and bd["x"] is not None:
                B[i] = ((bd["x"] + length / 2) / length, (bd["y"] + width / 2) / width)
            g = fr["possession"]["group"]
            lab10[i] = H if g == "home team" else A if g == "away team" else X
        n25 = int((n10 - 1) * FPS / SRC_FPS) + 1
        pl = np.concatenate([_upsample(P, n25), np.zeros((n25, len(players), 1), np.float32)], 2)
        ba = np.concatenate([_upsample(B, n25), np.full((n25, 1), np.nan, np.float32)], 1)
        idx = np.minimum(np.round(np.arange(n25) * SRC_FPS / FPS).astype(int), n10 - 1)
        if chunks_p:      # one ungraded separator frame between halves
            chunks_p.append(np.full((1, len(players), 3), np.nan, np.float32))
            chunks_b.append(np.full((1, 3), np.nan, np.float32))
            chunks_l.append(np.array([X], np.int8))
            period.append(np.array([0]))
        chunks_p.append(pl)
        chunks_b.append(ba)
        chunks_l.append(lab10[idx])
        period.append(np.full(n25, per))
    period = np.concatenate(period)
    blocks = []
    for per in (1, 2):
        idx = np.flatnonzero(period == per)
        if len(idx):
            blocks.append((int(idx[0]), int(idx[-1]) + 1))
    return {"name": "SC-" + mid, "players": np.concatenate(chunks_p),
            "ball": np.concatenate(chunks_b), "slot_team": slot_team,
            "labels": np.concatenate(chunks_l), "period": period, "blocks": blocks,
            "pitch": (length, width)}


def match_ids(root):
    return sorted(os.listdir(os.path.join(root, "data", "matches")))
