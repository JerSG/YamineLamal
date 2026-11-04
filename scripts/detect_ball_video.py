"""Optional step: detect the ball in the tactical video where neither tracker has it.

    pip install -r requirements-video.txt
    python scripts/detect_ball_video.py --gpu --limit 300   # quick trial (never used by run.py)
    python scripts/detect_ball_video.py --gpu               # NVIDIA GPU, deterministic fp32
    python scripts/detect_ball_video.py                     # CPU only: ~4 h
    python scripts/detect_ball_video.py --verify 300        # re-detect 300 frames, compare
    python run.py                                           # uses the result only if PASSED

Reproducibility: the detections are a frozen intermediate file whose SHA-256 is recorded in the
check file; run.py refuses a file that does not match, and turns the same file into exactly the
same submission on any machine. Re-running the detector itself gives identical detections on the
same machine and settings; on other hardware --verify reports how close it comes.

Needs data/tactical/BAR-ATH_tactical.mp4 in addition to the files run.py reads. Writes:

    data/derived/video_ball.csv          detected ball per frame (normalised pitch x, y, conf)
    data/derived/video_ball_raw.csv      every scanned frame's raw detection (pixels), for --verify
    data/derived/video_ball_check.json   the acceptance test, the versions used, the SHA-256

Both are derived from the match data: they are gitignored and must not be published.
The summary printed at the end contains only aggregate numbers, no match data.
"""

import argparse
import os

# Deterministic cuBLAS matrix maths; must be set before PyTorch starts CUDA.
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

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


RAW = os.path.join(DERIVED, "video_ball_raw.csv")


def write_raw(detections):
    with open(RAW, "w", newline="\n") as h:
        h.write("frame,x_px,y_px,conf\n")
        for f in sorted(detections):
            d = detections[f]
            h.write("{},,,\n".format(f) if d is None else
                    "{},{:.1f},{:.1f},{:.3f}\n".format(f, d[0], d[1], d[2]))


def verify(n, video_path, device, half, batch_frames):
    """Re-detect a fixed spread of n saved frames and compare with the saved raw detections."""
    if not os.path.isfile(RAW):
        sys.exit("nothing to verify: run the detection first ({} missing)".format(RAW))
    saved = {}
    with open(RAW) as h:
        next(h)
        for line in h:
            f, x, y, c = line.strip().split(",")
            saved[int(f)] = None if x == "" else (float(x), float(y), float(c))
    frames = sorted(saved)
    pick = [frames[i] for i in np.linspace(0, len(frames) - 1, min(n, len(frames))).astype(int)]
    detector = V.YoloBallDetector(device=device, half=half, batch_frames=batch_frames)
    print("verifying {} saved frames on {}{} ...".format(
        len(pick), detector.device, " (half precision)" if detector.half else ""))
    again = V.scan_video(video_path, pick, detector, games.source_frames("tactical"),
                         stop_after=len(pick))
    same = sum(again.get(f) == saved[f] for f in pick)
    close = sum((again.get(f) is None) == (saved[f] is None) and
                (saved[f] is None or np.hypot(again[f][0] - saved[f][0],
                                              again[f][1] - saved[f][1]) <= 2.0)
                for f in pick)
    print("\n=== verify (aggregate numbers only, safe to share) ===")
    print("identical detections: {}/{} ({:.1%})".format(same, len(pick), same / len(pick)))
    print("same ball within 2 px: {}/{} ({:.1%})".format(close, len(pick), close / len(pick)))
    print("RESULT: {}".format("IDENTICAL -- detections reproduce exactly" if same == len(pick)
                              else "NOT identical -- see the numbers above"))


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--gpu", action="store_true",
                    help="use the NVIDIA GPU (--device auto --batch-frames 8), full precision with "
                         "deterministic settings")
    ap.add_argument("--device", default=None,
                    help="cpu (default, reproducible), cuda:0, or auto (GPU if available)")
    ap.add_argument("--batch-frames", type=int, default=None,
                    help="frames per model call (default 1 on CPU, 8 with --gpu)")
    ap.add_argument("--fast", action="store_true",
                    help="half precision on the GPU: roughly 2x faster, slightly less repeatable "
                         "across machines")
    ap.add_argument("--verify", type=int, default=0, metavar="N",
                    help="re-detect N frames spread over the saved run and report how many match "
                         "data/derived/video_ball_raw.csv exactly (does not change any file)")
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
    device = a.device or ("auto" if a.gpu else "cpu")
    batch_frames = a.batch_frames or (8 if a.gpu else 1)
    half = a.gpu and a.fast

    paths = games.asset_paths()
    video_path = paths["video"]
    if a.verify:
        return verify(a.verify, video_path, device, half, batch_frames)
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

    detector = V.YoloBallDetector(device=device, half=half, batch_frames=batch_frames)
    print("detector: {} on {}{}, {} frame(s) per batch".format(
        C.VIDEO_WEIGHTS, detector.device, " (half precision)" if detector.half else "",
        detector.batch_frames))
    if detector.gpu:
        import torch
        print("  GPU: {}".format(torch.cuda.get_device_name(0)))
    elif a.gpu:
        print("  WARNING: --gpu given but PyTorch sees no CUDA GPU, so this runs on the CPU "
              "(slow).\n  Check: python -c \"import torch; print(torch.cuda.is_available())\"")
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
    meta = {"device": detector.device, "half_precision": detector.half,
            "batch_frames": detector.batch_frames,
            "threads": C.VIDEO_TORCH_THREADS, "weights": C.VIDEO_WEIGHTS,
            "torch": torch.__version__, "ultralytics": ultralytics.__version__,
            "python": platform.python_version(), "platform": platform.platform(),
            "seconds": round(time.time() - t0)}
    V.write_outputs(DERIVED, fill_points, stats, meta)
    write_raw(detections)

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
