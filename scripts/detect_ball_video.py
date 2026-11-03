"""Optional step: detect the ball in the tactical video where neither tracker has it.

    pip install -r requirements-video.txt
    python scripts/detect_ball_video.py              # CPU, deterministic; a few hours
    python scripts/detect_ball_video.py --limit 2000 # quick trial on the first 2000 wanted frames
    python run.py                                    # uses the result only if it PASSED

Needs data/tactical/BAR-ATH_tactical.mp4 in addition to the files run.py reads. Writes:

    data/derived/video_ball.csv          detected ball per frame (normalised pitch x, y, conf)
    data/derived/video_ball_check.json   the acceptance test and the versions used

Both are derived from the match data: they are gitignored and must not be published.
The summary printed at the end contains only aggregate numbers, no match data.
"""

import argparse
import os
import platform
import sys
import time

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import games  # noqa: E402
from possession import config as C  # noqa: E402
from possession import pipeline as P  # noqa: E402
from possession import sources as S  # noqa: E402
from possession import video as V  # noqa: E402

DERIVED = os.path.join(ROOT, "data", "derived")


def fused_ball(use_broadcast=True):
    """Tactical ATD ball, after the broadcast fill when the broadcast files are present."""
    paths = games.asset_paths()
    meta = S.read_atd_meta(paths["atd_meta"])
    n = games.source_frames("tactical")
    players, ball = S.read_atd(paths["atd_raw"], len(meta["slot_team"]), n)
    fused = ball
    hook = P.broadcast_hook(verbose=True) if use_broadcast else None
    if hook is not None:
        fused, _ = hook(players, ball, meta)
    return ball, fused


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--device", default="cpu",
                    help="cpu (default, reproducible) or e.g. cuda:0 (much faster; results can "
                         "differ slightly from a CPU run)")
    ap.add_argument("--threads", type=int, default=None,
                    help="CPU threads for inference (default {}: deterministic)".format(
                        C.VIDEO_TORCH_THREADS))
    ap.add_argument("--limit", type=int, default=0,
                    help="only process the first N wanted frames (a quick trial; never submit "
                         "from a trial run)")
    ap.add_argument("--no-broadcast", action="store_true")
    a = ap.parse_args()
    if a.threads:
        C.VIDEO_TORCH_THREADS = a.threads

    paths = games.asset_paths()
    video_path = paths["video"]
    if not os.path.isfile(video_path):
        sys.exit("video not found: {}".format(video_path))
    n = games.source_frames("tactical")
    pitch = games.GAMES[games.GAME]["pitch"]
    blocks = [(s, e) for _, s, e in P.graded_blocks()]

    t0 = time.time()
    tactical, fused = fused_ball(use_broadcast=not a.no_broadcast)
    targets, checks = V.select_frames(fused, tactical, blocks)
    wanted = np.union1d(targets, checks)
    if a.limit:
        wanted = wanted[:a.limit]
        targets = targets[np.isin(targets, wanted)]
        checks = checks[np.isin(checks, wanted)]
    print("frames to scan: {} missing-ball targets (every {}th) + {} check frames".format(
        len(targets), C.VIDEO_STEP, len(checks)))

    detector = V.YoloBallDetector(device=a.device)
    start = time.time()

    def progress(done, total, frame):
        rate = done / max(time.time() - start, 1e-9)
        print("  {:>6}/{} frames  ({:.1f} frames/s, ~{:.0f} min left)".format(
            done, total, rate, (total - done) / max(rate, 1e-9) / 60.0))
        sys.stdout.flush()

    detections = V.scan_video(video_path, wanted, detector, n, progress=progress,
                              stop_after=a.limit or None)
    H, ok = V.load_homographies(paths["homography"], n)
    points = V.drop_isolated_jumps(V.to_pitch(detections, H, ok, pitch), pitch)
    stats = V.check_against_tracker(points, tactical, checks, pitch)
    fill_points = {f: p for f, p in points.items() if f in set(int(t) for t in targets)}
    _, gap = V.fill_from_video(fused, fill_points, n, pitch)
    graded = np.zeros(n, dtype=bool)
    for s, e in blocks:
        graded[s:e] = True
    stats.update({
        "target_frames_scanned": int(len(targets)),
        "target_detections_kept": int(len(fill_points)),
        "graded_frames_filled": int((gap & graded).sum()),
        "trial_run": bool(a.limit),
    })
    if a.limit:
        stats["passed"] = False  # a trial never feeds the submission
    import torch
    import ultralytics
    meta = {"device": a.device, "threads": C.VIDEO_TORCH_THREADS, "weights": C.VIDEO_WEIGHTS,
            "torch": torch.__version__, "ultralytics": ultralytics.__version__,
            "python": platform.python_version(), "platform": platform.platform(),
            "seconds": round(time.time() - t0)}
    V.write_outputs(DERIVED, fill_points, stats, meta)

    print("\n=== video ball detection: summary (aggregate numbers only, safe to share) ===")
    print("check frames (tracker sees the ball): {check_frames}".format(**stats))
    print("  detector found the ball on {detected_on_check_frames} ({detection_rate:.1%})"
          .format(**stats))
    print("  median distance to the tracker's ball: {:.2f} m   within 3 m: {:.1%}".format(
        stats["median_error_m"], stats["share_within_3m"]))
    print("missing-ball frames scanned: {target_frames_scanned}, detections kept: "
          "{target_detections_kept}, graded frames filled: {graded_frames_filled}".format(**stats))
    print("RESULT: {}".format(
        "TRIAL ONLY (--limit) -- not used by run.py" if a.limit else
        "PASSED -- run.py will use it" if stats["passed"] else
        "FAILED -- run.py will ignore it; your submission is unchanged"))
    print("time: {:.0f} min".format((time.time() - t0) / 60.0))


if __name__ == "__main__":
    main()
