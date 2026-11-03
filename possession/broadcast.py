"""Align the broadcast recording to the tactical timeline, then use it to fill the tactical ball.

The broadcast feed is the same match filmed by a different camera, on its own timeline: cuts,
replays, zooms, and no frame mapping (DATA.md). Inside one continuous camera *shot*, though, both
recordings run at 25 fps, so a shot maps onto the tactical video by a single frame offset. The
offset is found by matching what both trackers see: the players.

    1. Shots. A shot is a maximal run of broadcast frames with a usable homography and no cut. A
       cut is a jump of the camera's aim point that no camera pan could make in 1/25 s.
    2. Offset per shot. For a few sample frames of the shot, the cost of every tactical frame t is
       the mean distance from each broadcast player to the nearest same-team tactical player
       (capped, so a partial view or a tracker error cannot dominate). Summing those costs at a
       common offset and taking the minimum gives the shot's offset. Every offset across the whole
       match is searched (coarse-to-fine), so replays find the moment they replay.
    3. Acceptance. A shot is kept only if its best offset fits well (players within ~1 m on
       average) and clearly beats every other offset. Then it is checked on a quantity the match
       did not use: where both trackers see the ball, the two ball positions must agree.

Everything is geometry on the provided tracking. No label is read.
"""

import numpy as np

from . import config as C
from .sources import HOME, AWAY


def camera_aim(homography_path, n_frames, image_size=(1920, 1080)):
    """Pitch point (normalised x, y) at the image centre per frame, and the valid flag."""
    data = np.genfromtxt(homography_path, delimiter=",", names=True)
    if len(data) != n_frames:
        raise ValueError("broadcast homography: {} rows, expected {}".format(len(data), n_frames))
    names = ["h%d%d" % (i, j) for i in range(3) for j in range(3)]
    M = np.stack([data[n] for n in names], axis=1).reshape(-1, 3, 3)
    ok = (data["valid"] == 1) & np.isfinite(M).all(axis=(1, 2))
    centre = np.array([image_size[0] / 2.0, image_size[1] / 2.0, 1.0])
    with np.errstate(all="ignore"):
        p = np.linalg.inv(np.where(ok[:, None, None], M, np.eye(3))) @ centre
        aim = np.stack([p[:, 0] / p[:, 2], p[:, 1] / p[:, 2]], axis=1)
    aim[~ok] = np.nan
    return aim, ok


def shots(aim, ok, n_players_seen, pitch):
    """[(start, end)] half-open runs of usable broadcast frames without a cut."""
    length, width = pitch
    step = np.hypot(np.diff(aim[:, 0]) * length, np.diff(aim[:, 1]) * width)
    usable = ok & (n_players_seen >= C.BC_MIN_PLAYERS)
    cut = np.zeros(len(ok), dtype=bool)
    cut[1:] = ~(step <= C.BC_CUT_JUMP_M)  # NaN step (an invalid neighbour) is a cut too
    out = []
    start = None
    for i in range(len(ok)):
        if usable[i] and start is not None and cut[i]:
            if i - start >= C.BC_MIN_SHOT_FRAMES:
                out.append((start, i))
            start = i
        elif usable[i] and start is None:
            start = i
        elif not usable[i] and start is not None:
            if i - start >= C.BC_MIN_SHOT_FRAMES:
                out.append((start, i))
            start = None
    if start is not None and len(ok) - start >= C.BC_MIN_SHOT_FRAMES:
        out.append((start, len(ok)))
    return out


class TacticalIndex(object):
    """Tactical player positions in metres, split by team, as float32 for fast scans."""

    def __init__(self, players, slot_team, pitch):
        length, width = pitch
        xy = players[..., :2].astype(np.float32) * np.array([length, width], dtype=np.float32)
        self.team = [xy[:, slot_team == HOME], xy[:, slot_team == AWAY]]
        self.n = len(players)

    def cost_at(self, points, teams, frames):
        """At the given tactical frames: mean over the points of the capped distance to the
        nearest same-team tactical player."""
        total = np.zeros(len(frames), dtype=np.float32)
        cap = np.float32(C.BC_MATCH_CAP_M)
        # Each team's positions at these frames are gathered once, not once per point. The
        # nearest player is found on squared distances and rooted once per frame: sqrt is
        # monotonic and correctly rounded, so this is exactly min(sqrt(d^2)), only cheaper.
        gathered = {side: (self.team[side][frames, :, 0], self.team[side][frames, :, 1])
                    for side in set(int(s) for s in teams)}
        for (x, y), side in zip(points, teams):
            tx, ty = gathered[int(side)]
            d2 = (tx - np.float32(x)) ** 2 + (ty - np.float32(y)) ** 2
            nearest = np.fmin.reduce(d2, axis=1)  # ignores NaN slots; NaN only if all are NaN
            d = np.sqrt(nearest)
            d[np.isnan(d)] = np.inf
            total += np.minimum(d, cap)
        return total / max(len(points), 1)


def align_shot(index, bc_players, bc_team, start, end, pitch):
    """Best tactical offset for one broadcast shot, its cost and its margin over the runner-up."""
    length, width = pitch
    seen = np.isfinite(bc_players[start:end, :, 0])
    rich = np.flatnonzero(seen.sum(axis=1) >= C.BC_MIN_PLAYERS)
    if len(rich) == 0:
        return None
    picks = rich[np.linspace(0, len(rich) - 1, min(C.BC_SAMPLES_PER_SHOT, len(rich))).astype(int)]
    samples, frames = [], []
    for k in picks:
        b = start + k
        idx = np.flatnonzero(seen[k])
        pts = [(bc_players[b, j, 0] * length, bc_players[b, j, 1] * width) for j in idx]
        samples.append((pts, bc_team[idx]))
        frames.append(b)
    frames = np.array(frames)
    lo, hi = -frames.min(), index.n - 1 - frames.max()
    if hi < lo:
        return None

    def total_cost(offsets):
        total = np.zeros(len(offsets), dtype=np.float32)
        for (pts, teams), b in zip(samples, frames):
            total += index.cost_at(pts, teams, b + offsets)
        return total / len(frames)

    # Coarse-to-fine: every BC_COARSE_STEP-th offset across the whole match, then every single
    # offset around the best few coarse minima. Players move well under a metre in 0.2 s, so the
    # true offset sits next to one of the lowest coarse points; refining several candidates
    # guards against near-ties such as a live shot and its replay.
    step = C.BC_COARSE_STEP
    coarse = np.arange(lo, hi + 1, step)
    c = total_cost(coarse)
    candidates = coarse[np.argsort(c, kind="stable")[:C.BC_REFINE_CANDIDATES]]
    fine = np.unique(np.concatenate([np.arange(o - step, o + step + 1) for o in candidates]))
    fine = fine[(fine >= lo) & (fine <= hi)]
    f = total_cost(fine)
    best = int(np.argmin(f))
    best_offset = int(fine[best])
    # Runner-up, refined at full resolution too, so the acceptance margin is never overstated by
    # the coarse grid: the best few coarse points outside the exclusion zone, each refined.
    far = np.abs(coarse - best_offset) > C.BC_RUNNER_UP_EXCLUSION
    if far.any():
        rivals = coarse[far][np.argsort(c[far], kind="stable")[:C.BC_REFINE_CANDIDATES]]
        around = np.unique(np.concatenate([np.arange(o - step, o + step + 1) for o in rivals]))
        around = around[(around >= lo) & (around <= hi) &
                        (np.abs(around - best_offset) > C.BC_RUNNER_UP_EXCLUSION)]
        runner_up = float(total_cost(around).min()) if len(around) else float("inf")
    else:
        runner_up = float("inf")
    return {"start": start, "end": end, "offset": best_offset, "cost_m": float(f[best]),
            "runner_up_m": runner_up, "n_samples": len(frames)}


_WORKER = {}


def _init_worker(tac_players, tac_slot_team, bc_players, bc_slot_team, pitch):
    _WORKER["index"] = TacticalIndex(tac_players, tac_slot_team, pitch)
    _WORKER["bc"] = (bc_players, bc_slot_team, pitch)


def _align_one(shot):
    bc_players, bc_slot_team, pitch = _WORKER["bc"]
    return align_shot(_WORKER["index"], bc_players, bc_slot_team, shot[0], shot[1], pitch)


def workers():
    """Processes for the offset search: BC_WORKERS, or one per CPU core (at most 8) when 0."""
    import os
    n = C.BC_WORKERS or min(8, os.cpu_count() or 1)
    return max(1, int(n))


def align(bc_players, bc_ball, bc_slot_team, tac_players, tac_ball, tac_slot_team, aim, ok,
          pitch, verbose=True):
    """Map broadcast frames to tactical frames. Returns (tac_of_bc, shot table).

    Shots are independent, so their offset searches run in parallel processes. A shot's result
    does not depend on which process computed it, so the output is identical for any number of
    workers.
    """
    n_bc = len(bc_players)
    n_seen = np.isfinite(bc_players[..., 0]).sum(axis=1)
    found = shots(aim, ok, n_seen, pitch)
    n_workers = workers()
    if verbose:
        print("  broadcast shots to align: {} ({} worker process{})".format(
            len(found), n_workers, "" if n_workers == 1 else "es"))
    if n_workers > 1 and len(found) > 1:
        import multiprocessing
        from concurrent.futures import ProcessPoolExecutor
        # "spawn" on every OS: it is the only mode Windows has, so the code path is the same
        # everywhere. Workers get positions only (x, y as float32): about half the memory.
        tac_xy = np.ascontiguousarray(tac_players[..., :2], dtype=np.float32)
        bc_xy = np.ascontiguousarray(bc_players[..., :2], dtype=np.float32)
        with ProcessPoolExecutor(max_workers=n_workers, initializer=_init_worker,
                                 mp_context=multiprocessing.get_context("spawn"),
                                 initargs=(tac_xy, tac_slot_team, bc_xy, bc_slot_team,
                                           pitch)) as pool:
            results = list(pool.map(_align_one, found, chunksize=1))
    else:
        index = TacticalIndex(tac_players, tac_slot_team, pitch)
        results = [align_shot(index, bc_players, bc_slot_team, s, e, pitch) for s, e in found]

    table = []
    tac_of_bc = np.full(n_bc, -1, dtype=np.int64)
    length, width = pitch
    for (s, e), r in zip(found, results):
        if r is None:
            continue
        r["accepted"] = (r["cost_m"] <= C.BC_MAX_COST_M and
                         r["runner_up_m"] - r["cost_m"] >= C.BC_MIN_MARGIN_M)
        # Independent check: the ball, which the offset search never looked at.
        t = np.arange(s, e) + r["offset"]
        inside = (t >= 0) & (t < len(tac_ball))
        both = inside.copy()
        both[inside] = (np.isfinite(bc_ball[s:e, 0])[inside] &
                        np.isfinite(tac_ball[t[inside], 0]))
        if both.any():
            d = np.hypot((bc_ball[s:e, 0][both] - tac_ball[t[both], 0]) * length,
                         (bc_ball[s:e, 1][both] - tac_ball[t[both], 1]) * width)
            r["ball_check_frames"] = int(both.sum())
            r["ball_median_err_m"] = float(np.median(d))
        else:
            r["ball_check_frames"] = 0
            r["ball_median_err_m"] = float("nan")
        if r["accepted"] and r["ball_check_frames"] >= C.BC_MIN_BALL_CHECK_FRAMES:
            r["accepted"] = r["ball_median_err_m"] <= C.BC_MAX_BALL_ERR_M
        if r["accepted"]:
            tac_of_bc[s:e][inside] = t[inside]
        table.append(r)
    return tac_of_bc, table


def fill_ball(tac_ball, bc_ball, tac_of_bc):
    """Tactical ball with gaps filled from aligned broadcast frames. Never overrides an observed
    tactical ball. Returns the filled array and a mask of the frames that were filled."""
    filled = tac_ball.copy()
    mask = np.zeros(len(tac_ball), dtype=bool)
    b = np.flatnonzero((tac_of_bc >= 0) & np.isfinite(bc_ball[:, 0]))
    t = tac_of_bc[b]
    gap = ~np.isfinite(tac_ball[t, 0])
    # A replay can map two broadcast frames onto one tactical frame: keep the first, which is
    # the live pass in broadcast order.
    t, keep = np.unique(t[gap], return_index=True)
    src = b[gap][keep]
    filled[t, :2] = bc_ball[src, :2]
    mask[t] = True
    return filled, mask
