"""Optional: find the ball in the tactical video where neither tracker has it.

    python scripts/detect_ball_video.py          # hours on a CPU; writes data/derived/
    python run.py                                # then uses the result if it passed its check

This is the one part of the pipeline that reads video, and it is OFF unless its output exists.
Nothing here ever overrides a ball that a tracker observed.

How it works
------------
1. Target frames: graded frames with no ball after the broadcast fill. To keep the run time sane
   the detector looks at every VIDEO_STEP-th target frame (frame % VIDEO_STEP == 0, so the choice
   is fixed). Gaps between two detections are filled by linear interpolation only when the two
   are close in time and in space (VIDEO_MAX_INTERP_*).
2. Check frames: a fixed sample of graded frames where the tactical tracker DOES see the ball.
   The detector runs there too, and its ball is compared with the tracker's. This is the
   acceptance test: it measures the detector on this very match with no labels involved.
3. Detection: a public pre-trained COCO detector (Ultralytics YOLO, class "sports ball"). The
   ball is ~10-15 px in the wide tactical view, too small for a single downscaled pass, so each
   frame is cut into overlapping tiles at native resolution.
4. Pixels -> pitch: the provided tactical homography maps pitch -> pixels; its inverse maps a
   detection back to normalised pitch coordinates. (A ball in the air projects to the point on
   the grass behind it; the acceptance test measures how much that costs.)
5. Clean-up: detections off the pitch are dropped, and so are isolated detections that would
   need an impossible ball speed to reach from both neighbours.

Reproducibility: inference is CPU-only and single-threaded by default with deterministic torch
settings, and coordinates and confidences are rounded before any decision, so that harmless
floating-point noise between machines cannot flip a decision. Different CPUs can still, rarely,
produce a different detection near the confidence threshold. The check file records the library
versions used, so a mismatch can be explained.
"""

import json
import os

import numpy as np

from . import config as C

TILE = 640           # detector input size; tiles are cut at native resolution
TILE_OVERLAP = 64    # pixels of overlap so a ball on a tile edge is seen whole


def tiles(width=1920, height=1080, size=TILE, overlap=TILE_OVERLAP):
    """[(x0, y0)] top-left corners of overlapping size x size tiles covering the image."""
    def starts(total):
        if total <= size:
            return [0]
        step = size - overlap
        out = list(range(0, total - size, step))
        out.append(total - size)
        return sorted(set(out))
    return [(x, y) for y in starts(height) for x in starts(width)]


def model_path():
    """data/models/<weights>: the detector weights live with the (gitignored) data, so the
    18 MB download can never be committed by accident."""
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    folder = os.path.join(root, "data", "models")
    os.makedirs(folder, exist_ok=True)
    return os.path.join(folder, C.VIDEO_WEIGHTS)


class YoloBallDetector(object):
    """Best 'sports ball' detection per frame, via tiled YOLO inference on the CPU."""

    COCO_SPORTS_BALL = 32

    def __init__(self, weights=None, device="cpu", conf=None):
        import torch
        from ultralytics import YOLO
        torch.use_deterministic_algorithms(True, warn_only=True)
        if device == "cpu":
            torch.set_num_threads(C.VIDEO_TORCH_THREADS)
        self.model = YOLO(weights or model_path())
        self.device = device
        self.conf = C.VIDEO_MIN_CONF if conf is None else conf

    def detect(self, frame_bgr):
        """(x_px, y_px, confidence) of the most confident ball, or None."""
        h, w = frame_bgr.shape[:2]
        corners = tiles(w, h)
        batch = [frame_bgr[y:y + TILE, x:x + TILE] for x, y in corners]
        results = self.model.predict(batch, imgsz=TILE, conf=self.conf, device=self.device,
                                     classes=[self.COCO_SPORTS_BALL], verbose=False)
        best = None
        for (x0, y0), r in zip(corners, results):
            if r.boxes is None or len(r.boxes) == 0:
                continue
            xywh = r.boxes.xywh.cpu().numpy()
            conf = r.boxes.conf.cpu().numpy()
            for (cx, cy, _, _), c in zip(xywh, conf):
                c = round(float(c), 3)
                cand = (round(float(cx) + x0, 1), round(float(cy) + y0, 1), c)
                if best is None or c > best[2]:
                    best = cand
        return best


def pixels_to_pitch(H, x_px, y_px):
    """Normalised pitch (x, y) of an image pixel, through the inverse of a pitch->pixel matrix."""
    p = np.linalg.inv(H) @ np.array([x_px, y_px, 1.0])
    return p[0] / p[2], p[1] / p[2]


def load_homographies(path, n_frames):
    data = np.genfromtxt(path, delimiter=",", names=True)
    if len(data) != n_frames:
        raise ValueError("homography: {} rows, expected {}".format(len(data), n_frames))
    names = ["h%d%d" % (i, j) for i in range(3) for j in range(3)]
    M = np.stack([data[n] for n in names], axis=1).reshape(-1, 3, 3)
    ok = (data["valid"] == 1) & np.isfinite(M).all(axis=(1, 2))
    return M, ok


def select_frames(fused_ball, tactical_ball, blocks):
    """(targets, checks): sorted arrays of video frames the detector should look at."""
    n = len(fused_ball)
    graded = np.zeros(n, dtype=bool)
    for s, e in blocks:
        graded[s:e] = True
    idx = np.arange(n)
    missing = graded & ~np.isfinite(fused_ball[:, 0])
    targets = idx[missing & (idx % C.VIDEO_STEP == 0)]
    seen = graded & np.isfinite(tactical_ball[:, 0])
    checks = idx[seen & (idx % C.VIDEO_CHECK_EVERY == 0)]
    return targets, checks


def scan_video(video_path, frames_wanted, detector, n_frames_expected, progress=None,
               stop_after=None):
    """Decode the video once, in order, and run the detector on the wanted frames only.

    Returns {frame: (x_px, y_px, conf) or None}. Raises if the video's frame count is not the
    reference count (FAQ: do not pad or trim to fit). stop_after (trial runs) stops early and
    skips the frame-count check.
    """
    import cv2
    wanted = set(int(f) for f in frames_wanted)
    out = {}
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise IOError("cannot open video: {}".format(video_path))
    frame = 0
    done = 0
    try:
        while True:
            if frame in wanted:
                ok, image = cap.read()
                if not ok:
                    break
                out[frame] = detector.detect(image)
                done += 1
                if progress and done % 500 == 0:
                    progress(done, len(wanted), frame)
                if stop_after and done >= stop_after:
                    return out
            else:
                if not cap.grab():
                    break
            frame += 1
    finally:
        cap.release()
    if frame != n_frames_expected:
        raise ValueError("decoded {} frames, expected {}: this is not the reference decode "
                         "(see FAQ.md); refusing to continue".format(frame, n_frames_expected))
    return out


def to_pitch(detections, H, ok, pitch):
    """{frame: (x_norm, y_norm, conf)} for detections that land on the pitch (with margin)."""
    length, width = pitch
    mx, my = C.VIDEO_PITCH_MARGIN_M / length, C.VIDEO_PITCH_MARGIN_M / width
    out = {}
    for f, det in detections.items():
        if det is None or not ok[f]:
            continue
        x, y = pixels_to_pitch(H[f], det[0], det[1])
        if -mx <= x <= 1 + mx and -my <= y <= 1 + my:
            out[f] = (round(float(x), 5), round(float(y), 5), det[2])
    return out


def drop_isolated_jumps(points, pitch):
    """Remove a detection when reaching it from both neighbouring detections would need a ball
    speed above VIDEO_MAX_BALL_SPEED_MS -- the signature of a false positive."""
    length, width = pitch
    frames = sorted(points)

    def speed(a, b):
        (xa, ya, _), (xb, yb, _) = points[a], points[b]
        return np.hypot((xb - xa) * length, (yb - ya) * width) * C.FPS / max(b - a, 1)

    fast = C.VIDEO_MAX_BALL_SPEED_MS

    def neighbours(seq, i):
        out = []
        for j in (i - 1, i + 1):
            if 0 <= j < len(seq) and abs(seq[j] - seq[i]) <= C.VIDEO_NEIGHBOUR_FRAMES:
                out.append(seq[j])
        return out

    # Pass 1, the odd one out: impossible to reach from both neighbours, while the two
    # neighbours agree with each other.
    survivors = []
    for i, f in enumerate(frames):
        near = neighbours(frames, i)
        if (len(near) == 2 and speed(near[0], f) > fast and speed(f, near[1]) > fast
                and speed(near[0], near[1]) <= fast):
            continue
        survivors.append(f)
    # Pass 2, lone contradictions: of two detections that cannot both be the ball and have no
    # third to arbitrate, keep the more confident (ties: the earlier frame, deterministically).
    keep = {}
    for i, f in enumerate(survivors):
        near = neighbours(survivors, i)
        bad = [g for g in near if speed(min(f, g), max(f, g)) > fast]
        if len(bad) == len(near) == 1:
            g = bad[0]
            if points[g][2] > points[f][2] or (points[g][2] == points[f][2] and g < f):
                continue
        keep[f] = points[f]
    return keep


def check_against_tracker(points, tactical_ball, checks, pitch):
    """Acceptance statistics on frames where the tracker sees the ball."""
    length, width = pitch
    found = [f for f in checks if f in points]
    errs = np.array([np.hypot((points[f][0] - tactical_ball[f, 0]) * length,
                              (points[f][1] - tactical_ball[f, 1]) * width) for f in found])
    n = len(checks)
    stats = {
        "check_frames": int(n),
        "detected_on_check_frames": int(len(found)),
        "detection_rate": float(len(found) / n) if n else 0.0,
        "median_error_m": float(np.median(errs)) if len(errs) else float("nan"),
        "share_within_3m": float(np.mean(errs <= 3.0)) if len(errs) else 0.0,
    }
    stats["passed"] = bool(
        len(found) >= C.VIDEO_MIN_CHECK_DETECTIONS
        and stats["median_error_m"] <= C.VIDEO_MAX_MEDIAN_ERR_M
        and stats["share_within_3m"] >= C.VIDEO_MIN_SHARE_WITHIN_3M)
    return stats


def interpolate(points, n_frames, pitch):
    """Per-frame (x, y) array: detections, plus straight-line fills between two detections that
    are at most VIDEO_MAX_INTERP_FRAMES apart and VIDEO_MAX_INTERP_M apart."""
    length, width = pitch
    xy = np.full((n_frames, 2), np.nan)
    frames = sorted(points)
    for f in frames:
        xy[f] = points[f][:2]
    for a, b in zip(frames[:-1], frames[1:]):
        if 1 < b - a <= C.VIDEO_MAX_INTERP_FRAMES:
            (xa, ya, _), (xb, yb, _) = points[a], points[b]
            if np.hypot((xb - xa) * length, (yb - ya) * width) <= C.VIDEO_MAX_INTERP_M:
                t = (np.arange(a + 1, b) - a) / float(b - a)
                xy[a + 1:b, 0] = xa + t * (xb - xa)
                xy[a + 1:b, 1] = ya + t * (yb - ya)
    return xy


# ------------------------------------------------------------------------------------------------
# files under data/derived/ (gitignored: derived from the match data, never published)
# ------------------------------------------------------------------------------------------------
def write_outputs(folder, points, stats, meta):
    os.makedirs(folder, exist_ok=True)
    with open(os.path.join(folder, "video_ball.csv"), "w", newline="\n") as h:
        h.write("frame,x,y,conf\n")
        for f in sorted(points):
            x, y, c = points[f]
            h.write("{},{:.5f},{:.5f},{:.3f}\n".format(f, x, y, c))
    with open(os.path.join(folder, "video_ball_check.json"), "w") as h:
        json.dump(dict(stats, **meta), h, indent=2, sort_keys=True)


def read_outputs(folder):
    """(points, stats) or (None, None) when the video step has not been run."""
    csv_path = os.path.join(folder, "video_ball.csv")
    json_path = os.path.join(folder, "video_ball_check.json")
    if not (os.path.isfile(csv_path) and os.path.isfile(json_path)):
        return None, None
    with open(json_path) as h:
        stats = json.load(h)
    points = {}
    with open(csv_path) as h:
        next(h)
        for line in h:
            f, x, y, c = line.strip().split(",")
            points[int(f)] = (float(x), float(y), float(c))
    return points, stats


def fill_from_video(ball, points, n_frames, pitch):
    """Ball array with remaining gaps filled from the video detections (never overrides)."""
    xy = interpolate(points, n_frames, pitch)
    filled = ball.copy()
    gap = ~np.isfinite(ball[:, 0]) & np.isfinite(xy[:, 0])
    filled[gap, 0] = xy[gap, 0]
    filled[gap, 1] = xy[gap, 1]
    return filled, gap
