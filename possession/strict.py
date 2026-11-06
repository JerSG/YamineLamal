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

PARAMS_VERSION = 2
TRACKING_FEATURES = ("ball", "pspeed", "bspeed", "bpos", "spread")


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


def tag_features(events, n_frames):
    """Smart Tagging features; their tables stay at the fixed priors in features.py, and set
    pieces are split into quarters of each tag (a fixed rule)."""
    feats = [F.phase_feature(events, n_frames),
             F.setpiece_feature(events, n_frames, fractions=SETPIECE_FRACTIONS),
             F.goal_feature(events, n_frames)]
    for f in feats:
        f.learn_team = f.learn_dead = False
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


def train(games, labels_list, verbose=True):
    """Fit tables + transitions, tune the scale; also a leave-one-game-out estimate."""
    summaries = [summarise(g["players"], g["ball"], g["slot_team"], C.STRICT_EXTERNAL_PITCH)
                 for g in games]
    bins = derive_bins(summaries, labels_list)
    if verbose:
        print("feature bins from the external games: {}".format(bins))
    with use_bins(bins):
        feats = [tracking_features(None, None, None, None, summary=t) for t in summaries]
    tables = fit_tables(feats, labels_list)
    trans, tstats = fit_transitions(labels_list)
    scale, curve = tune_scale(feats, labels_list, tables, trans)
    if verbose:
        print("scale search (mean over the 3 external games):")
        for s, sf, mf in curve:
            print("  scale {:>6.2f}  spellF1 {:.4f}  macroF1 {:.4f}{}".format(
                s, sf, mf, "   <- chosen" if s == scale else ""))
    loo = []
    for i in range(len(games)):
        tr = [j for j in range(len(games)) if j != i]
        # Everything, bins included, from the two training games only.
        t_bins = derive_bins([summaries[j] for j in tr], [labels_list[j] for j in tr])
        with use_bins(t_bins):
            t_feats = [tracking_features(None, None, None, None, summary=t) for t in summaries]
        t_tables = fit_tables([t_feats[j] for j in tr], [labels_list[j] for j in tr])
        t_trans, _ = fit_transitions([labels_list[j] for j in tr])
        t_scale, _ = tune_scale([t_feats[j] for j in tr], [labels_list[j] for j in tr],
                                t_tables, t_trans)
        sf, mf = spell_f1(labels_list[i], decode_game(t_feats[i], t_tables, t_trans, t_scale,
                                                      labels_list[i]))
        loo.append({"held_out": games[i]["name"], "scale": t_scale, "spell_f1": sf,
                    "macro_f1": mf})
        if verbose:
            print("leave-one-out: train on the other two, test on {}: spellF1 {:.4f} macroF1 "
                  "{:.4f} (scale {:.2f})".format(games[i]["name"], sf, mf, t_scale))
    names = TRACKING_FEATURES
    params = {
        "version": PARAMS_VERSION,
        "source": "Metrica Sports public sample games 1-3 (https://github.com/metrica-sports/"
                  "sample-data); labels derived mechanically from their event data",
        "bins": bins,
        "bins_rule": {"quantiles": BIN_QUANTILES, "spread_share": SPREAD_SHARE,
                      "setpiece_fractions": SETPIECE_FRACTIONS,
                      "control_radius_m": C.CONTROL_RADIUS_M},
        "features": {n: {"values": f.values, "table": t.tolist()}
                     for n, f, t in zip(names, feats[0], tables)},
        "transitions": trans.tolist(),
        "transition_stats": tstats,
        "scale": scale,
        "scale_curve": [{"scale": s, "spell_f1": a, "macro_f1": b} for s, a, b in curve],
        "leave_one_game_out": loo,
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


def apply(params, players, ball, slot_team, events, n_frames, blocks, pitch):
    """Labels H/A/D per frame (codes 0/1/2; -1 outside blocks) with the frozen parameters."""
    with use_bins(params["bins"]):
        tfeats = tracking_features(players, ball, slot_team, pitch)
    for f in tfeats:
        if f.values != params["features"][f.name]["values"]:
            raise ValueError("feature {} values differ from the trained parameters".format(f.name))
    feats = tfeats + tag_features(events, n_frames)
    tables = [np.array(params["features"][f.name]["table"]) for f in tfeats] + \
             [PossessionHMM._tables_from(f.init_team, f.init_dead, f) for f in feats[len(tfeats):]]
    model = FrozenHMM(feats, tables, params["transitions"], params["scale"], blocks)
    allowed, forced = kickoff_dead_mask(events, n_frames, blocks)
    return model.decode(blocks, allowed), model, forced
