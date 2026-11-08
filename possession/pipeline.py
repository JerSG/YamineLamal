"""Data in, possession file out (the strict-external model).

    competition tracking + Smart Tagging (+ broadcast)  ->  features  ->  frozen HMM
    (parameters from models/external_params.json, learned on other matches)  ->  Viterbi per
    half  ->  H / A / D on graded frames, X elsewhere

Broadcast fusion (on when data/broadcast/ holds the broadcast ATD + homography): the broadcast
recording is aligned to the tactical timeline shot by shot (possession/broadcast.py), and tactical
frames with no tracked ball take the ball from the aligned broadcast frame. An observed tactical
ball is never overridden. --no-broadcast switches it off.

Nothing here learns from the competition match: the model's parameters are read, frozen, from
models/external_params.json (scripts/train_external.py).
"""

import json
import os
import sys

import numpy as np

from . import config as C
from . import sources as S
from .hmm import STATE_LABELS

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
import games  # noqa: E402




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


STRICT_PARAMS = os.path.join(ROOT, "models", "external_params.json")


def run_strict(out_path=None, report_path=None, verbose=True, data_dir=None, use_broadcast=True,
               params_path=None):
    """The strict-external model (possession/strict.py): parameters learned on other matches,
    applied frozen. Same inputs and ball fusion as the default model; no fitting on this match."""
    from . import strict as ST
    params = ST.load(params_path or STRICT_PARAMS)
    paths = games.asset_paths(data_dir=data_dir)
    n_frames = games.source_frames("tactical")
    for k in ("atd_meta", "atd_raw", "smart_tagging"):
        if not S.file_exists(paths[k]):
            raise IOError("strict-external mode needs {}".format(paths[k]))
    hook = broadcast_hook(data_dir, verbose) if use_broadcast else None
    fusion = ["broadcast fusion"] if hook is not None else []
    if verbose:
        print("mode: strict-external (parameters from {}){}".format(
            os.path.relpath(params_path or STRICT_PARAMS, ROOT),
            "".join(" + " + f for f in fusion)))
    meta = S.read_atd_meta(paths["atd_meta"])
    if verbose:
        print("reading raw tracking {} ...".format(os.path.basename(paths["atd_raw"])))
    players, ball = S.read_atd(paths["atd_raw"], len(meta["slot_team"]), n_frames)
    info = {"mode": "strict-external", "ball_fusion": fusion}
    if hook is not None:
        ball, hook_info = hook(players, ball, meta)
        info.update(hook_info)
    events = S.read_smart_tagging(paths["smart_tagging"], C.FPS)
    halves = graded_blocks()
    blocks = [(s, e) for _, s, e in halves]
    path, model, forced = ST.apply(params, players, ball, meta["slot_team"], events, n_frames,
                                   blocks, games.GAMES[games.GAME]["pitch"])
    labels = np.full(n_frames, "X", dtype="<U1")
    graded = path >= 0
    labels[graded] = STATE_LABELS[path[graded]]
    stats = possession_stats(labels, blocks)
    info.update(stats)
    info["frames_forced_dead"] = int(forced.sum())
    info["external_scale"] = params["scale"]
    if verbose:
        print("output: {possessions} possessions (mean {mean_possession_s:.1f} s), "
              "{stoppages} stoppages; frames H {H} A {A} D {D}".format(**stats))
    if out_path:
        write_submission(out_path, labels)
        if verbose:
            print("wrote {}".format(out_path))
    if report_path:
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
