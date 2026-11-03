"""Data in, possession file out.

    assets  ->  per-frame categorical features  ->  tempered EM on the whole match (no labels)
            ->  Viterbi per half  ->  H / A / D on graded frames, X elsewhere

Modes
-----
  full   raw tracking present (data/tactical/BAR-ATH_tactical_atd.txt): the ball feature is built
         from player-to-ball distances, and kinematic features (player speed, ball speed, ball
         position against the lines, team spread) are added.
  basic  raw tracking absent: the ball feature is the organisers' published nearest-player signal
         (evaluation/results/baseline_raw.csv), which is the same quantity computed from the same
         ATD by starter/baseline.py. Smart Tagging and homography features are identical in both.
  auto   full if the raw tracking is present, else basic. The mode used is printed and recorded
         in the run report, so a file can always be regenerated with the same command.

Broadcast fusion (full mode, on when data/broadcast/ holds the broadcast ATD + homography): the
broadcast recording is aligned to the tactical timeline shot by shot (possession/broadcast.py), and
tactical frames with no tracked ball take the ball from the aligned broadcast frame. An observed
tactical ball is never overridden. --no-broadcast switches it off.
"""

import json
import os
import sys

import numpy as np

from . import config as C
from . import features as F
from . import sources as S
from .hmm import PossessionHMM, STATE_LABELS, D

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
import games  # noqa: E402

NEAREST_SIGNAL = os.path.join(ROOT, "evaluation", "results", "baseline_raw.csv")


def broadcast_hook(data_dir=None, verbose=True):
    """A ball_hook that fills tactical ball gaps from the aligned broadcast, or None if the
    broadcast files are not all present."""
    from . import broadcast as B
    bpaths = games.asset_paths(data_dir=data_dir, source="broadcast")
    need = ("atd_meta", "atd_raw", "homography")
    if not all(S.file_exists(bpaths[k]) for k in need):
        return None
    n_bc = games.source_frames("broadcast")

    def hook(players, ball, meta):
        if verbose:
            print("reading broadcast tracking + homography ...")
        bmeta = S.read_atd_meta(bpaths["atd_meta"])
        bp, bb = S.read_atd(bpaths["atd_raw"], len(bmeta["slot_team"]), n_bc)
        aim, ok = B.camera_aim(bpaths["homography"], n_bc)
        if verbose:
            print("aligning broadcast shots to the tactical timeline (about a minute) ...")
        tac_of_bc, table = B.align(bp, bb, bmeta["slot_team"], players, ball, meta["slot_team"],
                                   aim, ok, games.GAMES[games.GAME]["pitch"], verbose=verbose)
        filled, mask = B.fill_ball(ball, bb, tac_of_bc)
        acc = [r for r in table if r["accepted"]]
        checked = [r for r in acc if r["ball_check_frames"] > 0]
        errs = np.concatenate([[r["ball_median_err_m"]] * r["ball_check_frames"] for r in checked]) \
            if checked else np.array([np.nan])
        info = {"broadcast": {
            "shots": len(table), "shots_accepted": len(acc),
            "broadcast_frames_mapped": int((tac_of_bc >= 0).sum()),
            "tactical_ball_frames_filled": int(mask.sum()),
            "ball_check_median_error_m": float(np.median(errs)),
            "offsets_accepted": sorted({r["offset"] for r in acc}),
        }}
        if verbose:
            b = info["broadcast"]
            print("  shots accepted {}/{}; tactical ball filled on {} frames; independent ball "
                  "check: median {:.2f} m".format(b["shots_accepted"], b["shots"],
                                                  b["tactical_ball_frames_filled"],
                                                  b["ball_check_median_error_m"]))
        return filled, info
    return hook


def graded_blocks(game=games.GAME):
    """[(half, start, end)] half-open, in video frames."""
    return [(half, first, last + 1) for half, first, last in games.half_video_blocks(game)]


def home_attacks_right(n_frames, blocks, meta):
    """Per frame: does the home team attack towards x = 1? Halves swap ends."""
    first = (meta or {}).get("attack_direction_first_half", {}).get(S.HOME, "left_to_right")
    right_first = first.strip().lower() == "left_to_right"
    out = np.full(n_frames, right_first)
    for half, start, end in blocks:
        out[start:end] = right_first if half == 1 else not right_first
    return out


def dead_ball_constraints(events, n_frames, blocks):
    """Frames that can only be D: before each half's kickoff, and inside a kickoff tag before the
    ball is struck (kickoffs after goals included)."""
    allowed = np.ones((n_frames, 3), dtype=bool)
    forced = np.zeros(n_frames, dtype=bool)
    strikes = F.kickoff_strikes(events)
    for e in events:
        if e["code"] == "SET PIECES" and e["type"].startswith("kick off"):
            forced[max(0, e["start"]):min(n_frames, e["start"] + C.KICKOFF_STRIKE_OFFSET)] = True
    for half, start, end in blocks:
        inside = [s for s in strikes if start - C.KICKOFF_STRIKE_OFFSET <= s < end]
        if inside:
            forced[start:max(start, min(inside))] = True
    allowed[forced, 0] = False
    allowed[forced, 1] = False
    return allowed, forced


def build_features(mode, paths, events, n_frames, blocks, use_camera=True, verbose=True,
                   ball_hook=None):
    meta = S.read_atd_meta(paths["atd_meta"]) if S.file_exists(paths.get("atd_meta")) else None
    feats = []
    info = {"mode": mode, "broadcast_fusion": ball_hook is not None}

    if mode == "full":
        if meta is None:
            raise IOError("full mode needs the ATD metadata: {}".format(paths["atd_meta"]))
        if verbose:
            print("reading raw tracking {} ...".format(os.path.basename(paths["atd_raw"])))
        players, ball = S.read_atd(paths["atd_raw"], len(meta["slot_team"]), n_frames)
        if ball_hook is not None:  # e.g. fill tactical ball gaps from the aligned broadcast
            ball, hook_info = ball_hook(players, ball, meta)
            info.update(hook_info)
        t = F.tracking_summary(players, ball, meta["slot_team"], games.GAMES[games.GAME]["pitch"])
        del players
        info["ball_tracked_fraction"] = float(t["ball_seen"].mean())
        if S.file_exists(NEAREST_SIGNAL):
            ref = S.read_nearest_signal(NEAREST_SIGNAL, n_frames)
            agree = float(np.mean(ref == t["nearest"]))
            info["agreement_with_published_nearest_signal"] = agree
            if verbose:
                print("  nearest-team agreement with published baseline_raw: {:.4f}".format(agree))
        feats += [F.ball_feature_full(t), F.pspeed_feature(t), F.bspeed_feature(t),
                  F.bpos_feature(t), F.spread_feature(t)]
    else:
        if not S.file_exists(NEAREST_SIGNAL):
            raise IOError("basic mode needs {}".format(NEAREST_SIGNAL))
        nearest = S.read_nearest_signal(NEAREST_SIGNAL, n_frames)
        info["ball_tracked_fraction"] = float(np.mean(nearest >= 0))
        feats.append(F.ball_feature_basic(nearest))

    feats += [F.phase_feature(events, n_frames), F.setpiece_feature(events, n_frames),
              F.goal_feature(events, n_frames)]

    if use_camera and S.file_exists(paths.get("homography")):
        if verbose:
            print("reading homography ...")
        cam_x = F.camera_centre(paths["homography"], n_frames)
        feats.append(F.camzone_feature(cam_x, home_attacks_right(n_frames, blocks, meta)))
    info["features"] = [f.name for f in feats]
    return feats, info


def run(mode="auto", out_path=None, use_camera=True, report_path=None, verbose=True,
        data_dir=None, ball_hook=None, use_broadcast=True):
    paths = games.asset_paths(data_dir=data_dir)
    n_frames = games.source_frames("tactical")
    if not S.file_exists(paths["smart_tagging"]):
        raise IOError("Smart Tagging not found: {}".format(paths["smart_tagging"]))
    if mode == "auto":
        mode = "full" if S.file_exists(paths["atd_raw"]) else "basic"
    if ball_hook is None and use_broadcast and mode == "full":
        ball_hook = broadcast_hook(data_dir, verbose)
    if verbose:
        print("mode: {}{}".format(mode, " + broadcast fusion" if ball_hook is not None else ""))

    events = S.read_smart_tagging(paths["smart_tagging"], C.FPS)
    halves = graded_blocks()
    blocks = [(s, e) for _, s, e in halves]

    feats, info = build_features(mode, paths, events, n_frames, halves, use_camera, verbose,
                                 ball_hook)
    allowed, forced = dead_ball_constraints(events, n_frames, halves)
    if verbose:
        print("features: {}".format(", ".join(info["features"])))
        print("frames forced dead (pre-kickoff): {}".format(int(forced.sum())))
        print("fitting emissions by EM over {} graded frames ...".format(
            sum(e - s for s, e in blocks)))

    model = PossessionHMM(feats, blocks=blocks)
    history = []
    for round_ in range(C.MAX_CALIBRATION_ROUNDS):
        model.calibrate_scale(blocks, allowed, verbose=False)
        history.append(model.scale)
        if verbose:
            print("round {}: evidence scale {:.3f}".format(round_ + 1, model.scale))
        if len(history) > 1 and abs(history[-1] / history[-2] - 1.0) < C.SCALE_TOLERANCE:
            break
        model.fit(blocks, allowed, verbose=False)
    else:
        print("  warning: evidence scale did not settle in {} rounds: {}".format(
            C.MAX_CALIBRATION_ROUNDS, [round(h, 3) for h in history]))
    info["evidence_scale"] = model.scale
    info["evidence_scale_history"] = history
    path = model.decode(blocks, allowed)

    labels = np.full(n_frames, "X", dtype="<U1")
    graded = path >= 0
    labels[graded] = STATE_LABELS[path[graded]]

    stats = possession_stats(labels, blocks)
    info.update(stats)
    if verbose:
        print("\nlearned emission tables (H column; A is its home<->away mirror)")
        print(model.describe())
        print("\noutput: {possessions} possessions (mean {mean_possession_s:.1f} s), "
              "{stoppages} stoppages; frames H {H} A {A} D {D}".format(**stats))

    if out_path:
        write_submission(out_path, labels)
        if verbose:
            print("wrote {}".format(out_path))
    if report_path:
        # base = 1 / mean run length (capped at TEMPER); effective = base x evidence scale
        info["feature_base_weights"] = dict(zip([f.name for f in model.features],
                                                [float(w) for w in model.base_weights]))
        info["feature_effective_weights"] = dict(zip([f.name for f in model.features],
                                                     [float(w) for w in model.weights]))
        info["config"] = {k: getattr(C, k) for k in dir(C) if k.isupper()}
        info["emission_tables"] = {
            f.name: {v: {"H": round(float(t[0, i]), 6), "D": round(float(t[D, i]), 6)}
                     for i, v in enumerate(f.values)}
            for f, t in zip(model.features, model.tables)}
        with open(report_path, "w") as handle:
            json.dump(info, handle, indent=2, default=float)
    return labels, info


def possession_stats(labels, blocks):
    n_poss = n_dead = 0
    durations = []
    for s, e in blocks:
        seg = labels[s:e]
        change = np.flatnonzero(seg[1:] != seg[:-1]) + 1
        starts = np.concatenate(([0], change))
        ends = np.concatenate((change, [len(seg)]))
        for a, b in zip(starts, ends):
            if seg[a] in ("H", "A"):
                n_poss += 1
                durations.append((b - a) / float(C.FPS))
            elif seg[a] == "D":
                n_dead += 1
    counts = {k: int(np.sum(labels == k)) for k in "HADX"}
    out = {"possessions": n_poss, "stoppages": n_dead,
           "mean_possession_s": float(np.mean(durations)) if durations else 0.0}
    out.update(counts)
    return out


def write_submission(path, labels):
    folder = os.path.dirname(os.path.abspath(path))
    if not os.path.isdir(folder):
        os.makedirs(folder)
    with open(path, "w", newline="\n") as handle:
        handle.write("frame,label\n")
        handle.writelines("{},{}\n".format(i, l) for i, l in enumerate(labels))
