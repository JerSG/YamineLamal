"""Strict-external model: every learned or tuned parameter comes from OTHER matches.

    python scripts/train_external.py       # learns models/external_params.json from the public
                                           # Metrica sample games (see possession/external.py)
    python run.py --strict-external        # applies it, frozen, to the competition match

This answers the strict reading of the organisers' clarification ("If you need to train (or
retrain), you MUST use only external data. If your algorithm has parameters that need to be
tuned, you MUST tune them on external data"). Compared with the default model:

  learned on external games, then frozen   emission tables of the tracking features (supervised,
                                           from event-derived labels), the transition
                                           probabilities, and the evidence scale (chosen to
                                           maximise the organisers' spellF1 on those games)
  fixed rule, nothing estimated            feature weights: every run of identical feature values
                                           counts as one observation (weight 1 / run length,
                                           computed locally from the input itself)
  fixed priors, not learned                the Smart Tagging tables (no external game has tags)
  removed                                  the camera-zone feature (no external homography) and
                                           the kickoff offset measured on this match: a half is
                                           dead only before its kickoff tag opens

Nothing in this module reads the competition ground truth or video, and no parameter is fitted
to the competition match.
"""

import json
import os

import numpy as np

from . import config as C
from . import features as F
from .hmm import PossessionHMM, H, A, D

PARAMS_VERSION = 3
TRACKING_FEATURES = ("ball", "pspeed", "bspeed", "bpos", "spread")
# Added after the benchmark on external matches (scripts/benchmark.py, docs/TECHNICAL_REPORT.md):
# last touch (possession/touch.py), team shape (possession/shape.py), still ball
# and restart kick (possession/deadball.py). Each was kept because it raised the held-out spellF1.
EXTRA_FEATURES = ("lasttouch", "shape", "still", "restart")
MODEL_FEATURES = TRACKING_FEATURES + EXTRA_FEATURES


# ------------------------------------------------------------------------------------------------
# features, identical for external games and the competition match
# ------------------------------------------------------------------------------------------------
def speed_from_positions(players, pitch, half_window=None):
    """Player speed in m/s from positions (centred difference, smoothed). Used everywhere in
    strict mode so that external games (no speed channel) and the match are treated alike."""
    hw = C.STRICT_SPEED_SMOOTH_FRAMES if half_window is None else half_window
    length, width = pitch
    x = players[..., 0].astype(np.float64) * length
    y = players[..., 1].astype(np.float64) * width
    v = np.full(x.shape, np.nan)
    k = C.SPEED_CHORD_FRAMES
    if k and len(x) > 2 * k:
        v[k:-k] = np.hypot(x[2 * k:] - x[:-2 * k], y[2 * k:] - y[:-2 * k]) * C.FPS / (2 * k)
        with np.errstate(invalid="ignore"):
            v[v > C.MAX_PLAYER_SPEED_MS] = np.nan
    elif not k:
        v[1:] = np.hypot(np.diff(x, axis=0), np.diff(y, axis=0)) * C.FPS
    valid = np.isfinite(v)
    cs = np.concatenate([np.zeros((1,) + v.shape[1:]), np.cumsum(np.where(valid, v, 0), 0)])
    cn = np.concatenate([np.zeros((1,) + v.shape[1:]), np.cumsum(valid, 0)])
    idx = np.arange(len(v))
    lo, hi = np.maximum(0, idx - hw), np.minimum(len(v), idx + hw + 1)
    n = cn[hi] - cn[lo]
    with np.errstate(invalid="ignore", divide="ignore"):
        out = np.where(n > 0, (cs[hi] - cs[lo]) / np.maximum(n, 1), np.nan)
    out[~np.isfinite(players[..., 0])] = np.nan
    return out.astype(np.float32)


def summarise(players, ball, slot_team, pitch):
    """Physical per-frame quantities (features.tracking_summary), speeds from positions."""
    p = players.copy()
    p[..., 2] = speed_from_positions(players, pitch) / C.SPEED_UNIT_TO_MS
    return F.tracking_summary(p, ball, slot_team, pitch)


# Feature bin edges are not hand-picked in strict mode: they are equal-share quantiles of the
# external games (frames inside graded play only), so no number is chosen by looking at the
# competition match. The quantile levels themselves are a fixed rule, set before any data.
BIN_QUANTILES = {
    "GAP_BINS_S": (25, 50, 75),                 # ball-missing stretch length, frame-weighted
    "PLAYER_SPEED_BINS_MS": (20, 40, 60, 80),   # mean player speed
    "BALL_SPEED_BINS_MS": (25, 50, 75),         # ball speed where tracked
}
SPREAD_SHARE = 100.0 / 3                        # |log spread ratio| below which teams are "even"
SETPIECE_FRACTIONS = (0.25, 0.5, 0.75)          # quarters of each set-piece tag (no data)


def derive_bins(summaries, labels_list):
    """Bin edges from the external games' physical quantities."""
    gap_len, speed, bspeed, spread = [], [], [], []
    for t, lab in zip(summaries, labels_list):
        graded = lab != 3
        missing = t["nearest"] < 0
        gap_len.append(F.run_lengths(missing)[graded & missing] / float(C.FPS))
        speed.append(t["mean_speed"][graded])
        bspeed.append(t["ball_speed"][graded])
        spread.append(np.abs(t["log_spread"][graded]))
    def edges(xs, levels, name):
        x = np.concatenate(xs)
        x = x[np.isfinite(x)]
        if len(x) == 0:  # nothing measurable (e.g. too few tracked players): keep the default
            print("  warning: no external data for {}; keeping the default {}".format(
                name, getattr(C, name)))
            return getattr(C, name)
        return np.percentile(x, levels)

    bins = {
        "GAP_BINS_S": edges(gap_len, BIN_QUANTILES["GAP_BINS_S"], "GAP_BINS_S"),
        "PLAYER_SPEED_BINS_MS": edges(speed, BIN_QUANTILES["PLAYER_SPEED_BINS_MS"],
                                      "PLAYER_SPEED_BINS_MS"),
        "BALL_SPEED_BINS_MS": edges(bspeed, BIN_QUANTILES["BALL_SPEED_BINS_MS"],
                                    "BALL_SPEED_BINS_MS"),
        "SPREAD_LOG_RATIO": float(edges(spread, SPREAD_SHARE, "SPREAD_LOG_RATIO")),
    }
    return {k: (round(v, 4) if np.isscalar(v) else tuple(round(float(x), 4) for x in v))
            for k, v in bins.items()}


class use_bins(object):
    """Temporarily set the feature bin edges in config (restored on exit)."""

    def __init__(self, bins):
        self.bins = bins

    def __enter__(self):
        self.saved = {k: getattr(C, k) for k in self.bins}
        for k, v in self.bins.items():
            setattr(C, k, tuple(v) if isinstance(v, (list, tuple)) else v)

    def __exit__(self, *exc):
        for k, v in self.saved.items():
            setattr(C, k, v)


def tracking_features(players, ball, slot_team, pitch, summary=None):
    """The five tracking features, from positions only (speeds recomputed from positions), with
    whatever bin edges are currently set (see use_bins)."""
    t = summary if summary is not None else summarise(players, ball, slot_team, pitch)
    return [F.ball_feature_full(t), F.pspeed_feature(t), F.bspeed_feature(t), F.bpos_feature(t),
            F.spread_feature(t)]


def make_obs(players, ball, slot_team, pitch, blocks, labels=None, summary=None):
    """Everything the features read for one match: positions, their summary and graded blocks."""
    if summary is None:
        summary = summarise(players, ball, slot_team, pitch)
    return {"game": {"players": players, "ball": ball, "slot_team": np.asarray(slot_team)},
            "summary": summary, "blocks": blocks, "labels": labels, "pitch": pitch}


def model_features(o, shape_state):
    """The nine tracking features of the model, in MODEL_FEATURES order, with whatever bin
    edges are currently set (see use_bins)."""
    from . import deadball, shape, touch
    feats = tracking_features(None, None, None, None, summary=o["summary"])
    feats += touch.lasttouch_feature(o)
    feats += shape.shape_feature(o, shape_state)
    feats.append(deadball.still_feature(o["summary"]))
    feats.append(deadball.restart_feature(o["summary"]))
    assert tuple(f.name for f in feats) == MODEL_FEATURES
    return feats


def tag_features(events, n_frames):
    """Smart Tagging features; their tables stay at the fixed priors in features.py, and set
    pieces are split into quarters of each tag (a fixed rule)."""
    feats = [F.phase_feature(events, n_frames),
             F.setpiece_feature(events, n_frames, fractions=SETPIECE_FRACTIONS),
             F.goal_feature(events, n_frames)]
    return feats


def run_weights(codes, blocks, min_look_frames=None):
    """Per-frame weight 1 / max(run length, min_look_frames): each run of identical codes counts
    as one observation, but a feature can never add more than one observation per
    min_look_frames, however much it flickers. The cap keeps noisy tracking (many short runs)
    from flooding the model with evidence, which is what pushes it into fragmenting."""
    cap = C.STRICT_MIN_LOOK_FRAMES if min_look_frames is None else min_look_frames
    w = np.zeros(len(codes))
    for s, e in blocks:
        c = codes[s:e]
        if not len(c):
            continue
        change = np.flatnonzero(c[1:] != c[:-1]) + 1
        starts = np.concatenate(([0], change))
        ends = np.concatenate((change, [len(c)]))
        for a, b in zip(starts, ends):
            w[s + a:s + b] = 1.0 / max(b - a, cap)
    return w


def blocks_from_labels(lab):
    """[start, end) runs of graded (non-X) rows."""
    graded = lab != 3
    out, start = [], None
    for i, g in enumerate(graded):
        if g and start is None:
            start = i
        elif not g and start is not None:
            out.append((start, i))
            start = None
    if start is not None:
        out.append((start, len(lab)))
    return out


# ------------------------------------------------------------------------------------------------
# the frozen model
# ------------------------------------------------------------------------------------------------
class FrozenHMM(PossessionHMM):
    """PossessionHMM with given tables, transitions and per-frame (run-length) weights."""

    def __init__(self, features, tables, trans, scale, blocks):
        self.features = features
        self.tables = [np.asarray(t, dtype=float) for t in tables]
        self.trans = np.asarray(trans, dtype=float)
        self.log_trans = np.log(self.trans)
        self.scale = float(scale)
        self.frame_weights = [run_weights(f.codes, blocks) for f in features]
        self.base_weights = np.array([float(np.mean(w[w > 0])) if (w > 0).any() else 0.0
                                      for w in self.frame_weights])

    def log_emissions(self, allowed=None):
        T = len(self.features[0].codes)
        L = np.zeros((T, 3))
        for f, table, w in zip(self.features, self.tables, self.frame_weights):
            contribution = np.log(table[:, f.codes].T)
            contribution[f.neutral[f.codes]] = 0.0
            L += (self.scale * w)[:, None] * contribution
        if allowed is not None:
            L = np.where(allowed, L, -np.inf)
        return L


# ------------------------------------------------------------------------------------------------
# training on external games
# ------------------------------------------------------------------------------------------------
def fit_tables(per_game_feats, per_game_labels):
    """Supervised emission tables for the tracking features, tied exactly as in the default
    model (H/A mirror, D symmetric), with the same Dirichlet prior towards the starting tables."""
    tables = []
    for k in range(len(per_game_feats[0])):
        f0 = per_game_feats[0][k]
        V = len(f0.values)
        cH, cA, cD = np.zeros(V), np.zeros(V), np.zeros(V)
        for feats, lab in zip(per_game_feats, per_game_labels):
            codes = feats[k].codes
            cH += np.bincount(codes[lab == H], minlength=V)
            cA += np.bincount(codes[lab == A], minlength=V)
            cD += np.bincount(codes[lab == D], minlength=V)
        prior_team = np.where(f0.neutral, 0.0, f0.init_team)
        prior_dead = np.where(f0.neutral, 0.0, f0.init_dead)
        team = cH + cA[f0.swap] + C.PRIOR_STRENGTH * prior_team
        dead = cD + C.PRIOR_STRENGTH * prior_dead
        tables.append(PossessionHMM._tables_from(team, dead, f0))
    return tables


def fit_transitions(per_game_labels):
    """Per-frame transition matrix from the labels' own spells (pooled over games)."""
    alive_frames = dead_frames = 0
    alive_to_other = alive_to_dead = dead_exits = 0
    for lab in per_game_labels:
        for s, e in blocks_from_labels(lab):
            seq = lab[s:e]
            alive_frames += int(np.sum(seq != D))
            dead_frames += int(np.sum(seq == D))
            for a, b in zip(seq[:-1], seq[1:]):
                if a == b:
                    continue
                if a == D:
                    dead_exits += 1
                elif b == D:
                    alive_to_dead += 1
                else:
                    alive_to_other += 1
    p_other = alive_to_other / float(alive_frames)
    p_dead = alive_to_dead / float(alive_frames)
    p_exit = dead_exits / float(dead_frames)
    trans = np.array([[1 - p_other - p_dead, p_other, p_dead],
                      [p_other, 1 - p_other - p_dead, p_dead],
                      [p_exit / 2, p_exit / 2, 1 - p_exit]])
    stats = {"mean_alive_spell_s": 1.0 / (p_other + p_dead) / C.FPS,
             "share_of_spells_ending_in_stoppage": p_dead / (p_other + p_dead),
             "mean_dead_spell_s": 1.0 / p_exit / C.FPS}
    return trans, stats


def decode_game(feats, tables, trans, scale, lab):
    blocks = blocks_from_labels(lab)
    model = FrozenHMM(feats, tables, trans, scale, blocks)
    path = model.decode(blocks)
    out = np.full(len(lab), 3, dtype=np.int8)
    out[path >= 0] = path[path >= 0]
    return out


def spell_f1(lab, pred):
    import sys
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    sys.path.insert(0, os.path.join(root, "evaluation"))
    import possession_metrics as pm
    r = pm.evaluate(lab.astype(int), pred.astype(int))
    return float(r["spell_f1_mean"]), float(r["macro_f1"])


def tune_scale(per_game_feats, per_game_labels, tables, trans, grid=None):
    """The evidence scale with the best mean spellF1 over the given external games."""
    grid = C.STRICT_SCALE_GRID if grid is None else grid
    rows = []
    for scale in grid:
        scores = [spell_f1(lab, decode_game(feats, tables, trans, scale, lab))
                  for feats, lab in zip(per_game_feats, per_game_labels)]
        rows.append((scale, float(np.mean([s[0] for s in scores])),
                     float(np.mean([s[1] for s in scores]))))
    best = max(rows, key=lambda r: (r[1], -r[0]))
    return best[0], rows


def train(games, labels_list, source, verbose=True):
    """Fit the team-shape model, the emission tables and the transitions on the external games,
    and tune the evidence scale on them. Held-out evaluation is scripts/benchmark.py."""
    from . import shape
    obs = [make_obs(g["players"], g["ball"], g["slot_team"], C.STRICT_EXTERNAL_PITCH,
                    blocks_from_labels(lab), labels=lab) for g, lab in zip(games, labels_list)]
    bins = derive_bins([o["summary"] for o in obs], labels_list)
    if verbose:
        print("feature bins from the external games: {}".format(bins))
    shape_state = shape.fit(obs)
    if verbose:
        print("team-shape model: training accuracy {:.3f} on live frames".format(
            shape_state["train_accuracy"]))
    with use_bins(bins):
        feats = [model_features(o, shape_state) for o in obs]
    tables = fit_tables(feats, labels_list)
    trans, tstats = fit_transitions(labels_list)
    scale, curve = tune_scale(feats, labels_list, tables, trans)
    if verbose:
        print("scale search (mean over the {} external games):".format(len(games)))
        for s_, sf, mf in curve:
            print("  scale {:>6.2f}  spellF1 {:.4f}  macroF1 {:.4f}{}".format(
                s_, sf, mf, "   <- chosen" if s_ == scale else ""))
    params = {
        "version": PARAMS_VERSION,
        "source": source,
        "training_games": [g["name"] for g in games],
        "bins": bins,
        "bins_rule": {"quantiles": BIN_QUANTILES, "spread_share": SPREAD_SHARE,
                      "setpiece_fractions": SETPIECE_FRACTIONS,
                      "control_radius_m": C.CONTROL_RADIUS_M},
        "shape_model": {"weights": [round(float(x), 8) for x in shape_state["w"]],
                        "column_scale": [round(float(x), 8) for x in shape_state["scale"]],
                        "train_accuracy": round(shape_state["train_accuracy"], 4)},
        "features": {n: {"values": f.values, "table": t.tolist()}
                     for n, f, t in zip(MODEL_FEATURES, feats[0], tables)},
        "speed_chord_frames": C.SPEED_CHORD_FRAMES,
        "transitions": trans.tolist(),
        "transition_stats": tstats,
        "scale": scale,
        "scale_curve": [{"scale": s_, "spell_f1": a, "macro_f1": b} for s_, a, b in curve],
    }
    return params


def save(params, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as h:
        json.dump(params, h, indent=2, sort_keys=True)


def load(path):
    with open(path) as h:
        params = json.load(h)
    if params.get("version") != PARAMS_VERSION:
        raise ValueError("{}: parameters version {} != {}".format(
            path, params.get("version"), PARAMS_VERSION))
    return params


# ------------------------------------------------------------------------------------------------
# applying the frozen model to the competition match
# ------------------------------------------------------------------------------------------------
def kickoff_dead_mask(events, n_frames, blocks):
    """Frames of a graded half before its kickoff tag opens are dead. No offset measured on the
    match: the tag's own start is the boundary (the model decides inside the tag)."""
    starts = sorted(e["start"] for e in events
                    if e["code"] == "SET PIECES" and e["type"].startswith("kick off (start)"))
    forced = np.zeros(n_frames, dtype=bool)
    for s, e in blocks:
        inside = [k for k in starts if s <= k < e]
        if inside:
            forced[s:min(inside)] = True
    allowed = np.ones((n_frames, 3), dtype=bool)
    allowed[forced, 0] = allowed[forced, 1] = False
    return allowed, forced


GOAL_KICKOFF_MAX_S = 180.0   # [football] a goal's kickoff comes within 3 minutes (celebration, VAR)


def goal_dead_mask(events, n_frames, blocks):
    """After a goal the ball is dead until the kickoff that restarts play (Laws of the Game).
    Conservative, so it never cuts into the move that scored: frames from the END of a GOALS tag
    to the start of the next "Kick Off (After Goal)" tag can only be dead, when that kickoff opens
    within GOAL_KICKOFF_MAX_S of the goal tag's end. A goal tag with no such kickoff (a
    disallowed goal, for instance) forces nothing. Fixed rule, no offset measured on any match."""
    kickoffs = np.array(sorted(e["start"] for e in events if e["code"] == "SET PIECES"
                               and e["type"].startswith("kick off (after goal)")))
    forced = np.zeros(n_frames, dtype=bool)
    graded = np.zeros(n_frames, dtype=bool)
    for s, e in blocks:
        graded[s:e] = True
    for ev in events:
        if ev["code"] != "GOALS" or not len(kickoffs):
            continue
        start = max(0, ev["end"])
        after = kickoffs[kickoffs >= start]
        if len(after) and after[0] - start <= GOAL_KICKOFF_MAX_S * C.FPS:
            forced[start:min(int(after[0]), n_frames)] = True
    return forced & graded


def apply(params, players, ball, slot_team, events, n_frames, blocks, pitch):
    """Labels H/A/D per frame (codes 0/1/2; -1 outside blocks) with the frozen parameters."""
    sm = params["shape_model"]
    shape_state = {"w": np.array(sm["weights"]), "scale": np.array(sm["column_scale"])}
    saved_chord = C.SPEED_CHORD_FRAMES
    C.SPEED_CHORD_FRAMES = params.get("speed_chord_frames", 0)
    try:
        o = make_obs(players, ball, slot_team, pitch, blocks)
    finally:
        C.SPEED_CHORD_FRAMES = saved_chord
    with use_bins(params["bins"]):
        tfeats = model_features(o, shape_state)
    for f in tfeats:
        if f.values != params["features"][f.name]["values"]:
            raise ValueError("feature {} values differ from the trained parameters".format(f.name))
    feats = tfeats + tag_features(events, n_frames)
    tables = [np.array(params["features"][f.name]["table"]) for f in tfeats] + \
             [PossessionHMM._tables_from(f.init_team, f.init_dead, f) for f in feats[len(tfeats):]]
    model = FrozenHMM(feats, tables, params["transitions"], params["scale"], blocks)
    allowed, forced = kickoff_dead_mask(events, n_frames, blocks)
    after_goal = goal_dead_mask(events, n_frames, blocks)
    allowed[after_goal, 0] = allowed[after_goal, 1] = False
    forced = forced | after_goal
    model.frames_dead_after_goals = int(after_goal.sum())
    return model.decode(blocks, allowed), model, forced
