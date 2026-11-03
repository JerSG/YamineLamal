"""Per-frame categorical observations for the HMM.

Every feature is a small categorical variable per frame. Team-bearing features are written in
*absolute* terms (home / away); the HMM ties the H and A states by swapping home<->away, so a
feature means the same thing for both teams and the model has half the parameters to learn.

Each feature carries the *starting point* of its emission table for the possessing-team state
(written from the home team's point of view) and for the dead-ball state. EM moves both, on the
whole match, without labels -- see hmm.py. The starting points are deliberately mild.

  basic features (need only the nearest-team signal + Smart Tagging)
    ball        nearest team to the ball, or length of the ball-missing stretch
    phase       open-play Smart Tagging phases, translated into "which team is attacking"
    setpiece    inside a set-piece tag: team that takes it x time since the tag opened
    goal        inside a goal tag: time left until the restart kickoff
  full features (need the raw tracking)
    ball        as above, but split by whether the nearest player is on the ball (< 2 m) and
                with a "contested" value when both teams are on it
    pspeed      mean speed of the tracked players (stoppages are walked)
    bpos        ball position against the touch/goal lines, and "lost after going out"
    bspeed      ball speed
    spread      which team is more spread out (the team in possession opens the pitch up)
"""

import numpy as np

from . import config as C
from .sources import HOME, AWAY


class Feature(object):
    def __init__(self, name, values, codes, init_team, init_dead, team_bearing=True,
                 learn_team=True, learn_dead=True, neutral=()):
        self.name = name
        # Values that carry no evidence ("not measurable on this frame"). They contribute a
        # likelihood ratio of 1 to every state, so the *absence* of a measurement can never be
        # learned as evidence by this feature -- missingness is modelled once, by the ball feature.
        self.neutral = np.array([v in neutral for v in values], dtype=bool)
        # Which of the two emission tables EM may move. A table that is not learned stays at its
        # starting point. See hmm.py, "identifiability", for why the dead-ball table is learned
        # only for features that physically measure a dead ball.
        self.learn_team = learn_team
        self.learn_dead = learn_dead
        self.values = list(values)
        self.codes = np.asarray(codes, dtype=np.int16)
        self.team_bearing = team_bearing
        self.swap = np.array([self.values.index(_swap_name(v)) for v in self.values]) \
            if team_bearing else np.arange(len(self.values))
        missing = [v for v in self.values
                   if v not in neutral and (v not in init_team or v not in init_dead)]
        if missing:
            raise ValueError("feature {}: no starting probability for {}".format(name, missing))
        self.init_team = self.renormalise([init_team.get(v, 0.0) for v in self.values])
        self.init_dead = self.renormalise([init_dead.get(v, 0.0) for v in self.values])
        if self.codes.min() < 0 or self.codes.max() >= len(self.values):
            raise ValueError("feature {}: code out of range".format(name))

    def renormalise(self, weights):
        """Normalise over the evidence-bearing values; neutral entries are set to 1 (ignored)."""
        w = np.where(self.neutral, 0.0, np.asarray(weights, dtype=float))
        w = w / w.sum()
        return np.where(self.neutral, 1.0, w)

    def histogram(self, mask):
        counts = np.bincount(self.codes[mask], minlength=len(self.values))
        return dict(zip(self.values, counts.tolist()))


def _swap_name(value):
    if "home" in value:
        return value.replace("home", "away")
    if "away" in value:
        return value.replace("away", "home")
    return value


def _normalise(weights):
    w = np.asarray(weights, dtype=float)
    return w / w.sum()


# ------------------------------------------------------------------------------------------------
# helpers
# ------------------------------------------------------------------------------------------------
def run_lengths(mask):
    """For every frame, the length of the run of equal mask values it sits in."""
    n = len(mask)
    out = np.zeros(n, dtype=np.int64)
    if n == 0:
        return out
    change = np.flatnonzero(np.diff(mask.astype(np.int8))) + 1
    starts = np.concatenate(([0], change))
    ends = np.concatenate((change, [n]))
    for s, e in zip(starts, ends):
        out[s:e] = e - s
    return out


def gap_bin(missing):
    """0..len(GAP_BINS_S) for missing frames, by the length of the missing stretch."""
    length_s = run_lengths(missing) / float(C.FPS)
    return np.searchsorted(np.asarray(C.GAP_BINS_S), length_s, side="right")


GAP_VALUES = ["gap<1s", "gap1-4s", "gap4-10s", "gap>10s"]


def centred_mean(x, half_window):
    """NaN-aware centred moving mean."""
    valid = np.isfinite(x)
    v = np.where(valid, x, 0.0)
    cs = np.concatenate(([0.0], np.cumsum(v)))
    cn = np.concatenate(([0], np.cumsum(valid)))
    idx = np.arange(len(x))
    lo = np.maximum(0, idx - half_window)
    hi = np.minimum(len(x), idx + half_window + 1)
    n = cn[hi] - cn[lo]
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(n > 0, (cs[hi] - cs[lo]) / np.maximum(n, 1), np.nan)


# ------------------------------------------------------------------------------------------------
# basic features
# ------------------------------------------------------------------------------------------------
def ball_feature_basic(nearest):
    """nearest: per-frame 0 home / 1 away / -1 ball not tracked."""
    values = ["home", "away"] + GAP_VALUES
    missing = nearest < 0
    codes = np.where(missing, 2 + gap_bin(missing), nearest).astype(np.int16)
    init_team = {"home": 0.42, "away": 0.18, "gap<1s": 0.06, "gap1-4s": 0.08,
                 "gap4-10s": 0.12, "gap>10s": 0.14}
    init_dead = {"home": 0.15, "away": 0.15, "gap<1s": 0.05, "gap1-4s": 0.10,
                 "gap4-10s": 0.20, "gap>10s": 0.35}
    return Feature("ball", values, codes, init_team, init_dead, learn_dead=False)


# Open-play phases: the tag's Team is the team in possession ...
ATTACKING_CODES = ("BUILD UP", "PROGRESSION", "CREATING CHANCES", "ATTACKING TRANSITION",
                   "BALL IN FINAL THIRD", "BALL IN THE BOX", "PLAYERS IN THE BOX", "LONG BALL")
# ... or the team out of possession.
DEFENDING_CODES = ("DEFENDING IN ATTACKING THIRD", "DEFENDING IN MIDDLE THIRD",
                   "DEFENDING IN DEFENSIVE THIRD", "DEFENSIVE TRANSITION")


def phase_feature(events, n_frames):
    votes = np.zeros((n_frames, 2), dtype=np.int16)
    for e in events:
        if e["team"] is None:
            continue
        if e["code"] in ATTACKING_CODES:
            side = e["team"]
        elif e["code"] in DEFENDING_CODES:
            side = 1 - e["team"]
        else:
            continue
        s, t = max(0, e["start"]), min(n_frames, e["end"])
        votes[s:t, side] += 1
    h, a = votes[:, 0] > 0, votes[:, 1] > 0
    codes = np.select([h & a, h, a], [3, 1, 2], 0)
    values = ["none", "home", "away", "both"]
    init_team = {"none": 0.40, "home": 0.45, "away": 0.10, "both": 0.05}
    init_dead = {"none": 0.70, "home": 0.13, "away": 0.13, "both": 0.04}
    return Feature("phase", values, codes, init_team, init_dead, learn_dead=False)


def kickoff_strikes(events):
    """Video frames at which a kickoff (start of half or after a goal) is struck."""
    return sorted(e["start"] + C.KICKOFF_STRIKE_OFFSET for e in events
                  if e["code"] == "SET PIECES" and e["type"].startswith("kick off"))


def setpiece_feature(events, n_frames):
    """Inside a (non-kickoff) set-piece tag: taker x seconds since the tag opened."""
    bins = np.asarray(C.SETPIECE_BINS_S) * C.FPS
    n_bins = len(bins) + 1
    teams = ["home", "away", "na"]
    values = ["none"] + ["{}_t{}".format(t, b) for t in teams for b in range(n_bins)]
    codes = np.zeros(n_frames, dtype=np.int16)
    for e in events:  # sorted by start, so a later set piece overrides an earlier overlap
        if e["code"] != "SET PIECES" or e["type"].startswith("kick off"):
            continue
        s, t = max(0, e["start"]), min(n_frames, e["end"])
        if t <= s:
            continue
        team = {HOME: 0, AWAY: 1, None: 2}[e["team"]]
        b = np.searchsorted(bins, np.arange(t - s), side="right")
        codes[s:t] = 1 + team * n_bins + b
    # Starting point: early in the window it is more likely dead; later the taker has the ball.
    init_team, init_dead = {"none": 0.9}, {"none": 0.8}
    for b in range(n_bins):
        late = b / float(n_bins - 1)
        init_team["home_t%d" % b] = 0.004 + 0.012 * late
        init_team["away_t%d" % b] = 0.004 + 0.004 * late
        init_team["na_t%d" % b] = 0.004 + 0.006 * late
        init_dead["home_t%d" % b] = init_dead["away_t%d" % b] = 0.016 - 0.008 * late
        init_dead["na_t%d" % b] = 0.016 - 0.008 * late
    return Feature("setpiece", values, codes, init_team, init_dead, learn_dead=False)


def goal_feature(events, n_frames):
    """Inside a goal tag, before its kickoff strike: seconds left until the restart, binned."""
    strikes = np.array(kickoff_strikes(events))
    edges = np.asarray(C.GOAL_BINS_S) * C.FPS
    values = ["none"] + ["ko<{:.0f}s".format(edges[0] / C.FPS)] + \
        ["ko{:.0f}-{:.0f}s".format(a / C.FPS, b / C.FPS) for a, b in zip(edges[:-1], edges[1:])] + \
        [">{:.0f}s".format(edges[-1] / C.FPS)]
    codes = np.zeros(n_frames, dtype=np.int16)
    for e in events:
        if e["code"] != "GOALS":
            continue
        after = strikes[strikes > e["start"]]
        if not len(after):
            continue
        ko = int(min(after[0], n_frames))
        s = max(0, e["start"])
        if ko <= s:
            continue
        left = ko - np.arange(s, ko)
        codes[s:ko] = 1 + np.searchsorted(edges, left, side="left")
    # Starting point: a goal tag's last two minutes are mostly celebration and restart.
    init_team = dict(zip(values, [0.970, 0.003, 0.004, 0.008, 0.015]))
    init_dead = dict(zip(values, [0.850, 0.050, 0.040, 0.035, 0.025]))
    return Feature("goal", values, codes, init_team, init_dead, team_bearing=False,
                   learn_team=False, learn_dead=False)


# ------------------------------------------------------------------------------------------------
# full features (raw tracking)
# ------------------------------------------------------------------------------------------------
def tracking_summary(players, ball, slot_team, pitch):
    """Per-frame physical quantities, in metres and m/s, from the raw ATD arrays."""
    length, width = pitch
    px, py = players[..., 0] * length, players[..., 1] * width
    ps = players[..., 2] * C.SPEED_UNIT_TO_MS
    bx, by = ball[:, 0] * length, ball[:, 1] * width
    seen = np.isfinite(px) & np.isfinite(py)
    ball_seen = np.isfinite(bx) & np.isfinite(by)

    d = np.sqrt((px - bx[:, None]) ** 2 + (py - by[:, None]) ** 2)
    d = np.where(seen, d, np.inf)
    home_slots, away_slots = slot_team == HOME, slot_team == AWAY
    with np.errstate(invalid="ignore"):
        dh = d[:, home_slots].min(axis=1) if home_slots.any() else np.full(len(d), np.inf)
        da = d[:, away_slots].min(axis=1) if away_slots.any() else np.full(len(d), np.inf)

    nearest = np.where(~ball_seen | ~np.isfinite(np.minimum(dh, da)), -1,
                       np.where(dh <= da, HOME, AWAY)).astype(np.int8)
    # Tie-break identical to starter/baseline.py (strict <, first slot wins) is not reproduced
    # exactly on equal distances; those are vanishingly rare with float positions.

    n_seen = seen.sum(axis=1)
    speed_sum = np.where(seen, np.nan_to_num(ps), 0.0).sum(axis=1)
    speed_n = (seen & np.isfinite(ps)).sum(axis=1)
    with np.errstate(invalid="ignore", divide="ignore"):
        mean_speed = np.where((n_seen >= C.MIN_TRACKED_PLAYERS) & (speed_n > 0),
                              speed_sum / np.maximum(speed_n, 1), np.nan)

    # Ball speed from positions, only between two consecutive tracked frames.
    v = np.full(len(bx), np.nan)
    step = np.hypot(np.diff(bx), np.diff(by)) * C.FPS
    both = ball_seen[1:] & ball_seen[:-1]
    v[1:] = np.where(both, step, np.nan)
    ball_speed = centred_mean(v, C.BALL_SPEED_SMOOTH_FRAMES)
    ball_speed[~ball_seen] = np.nan

    def spread(slots):
        x = np.where(seen[:, slots], px[:, slots], np.nan)
        y = np.where(seen[:, slots], py[:, slots], np.nan)
        with np.errstate(invalid="ignore"):
            n = np.isfinite(x).sum(axis=1)
            area = np.nanstd(x, axis=1) * np.nanstd(y, axis=1)
        return np.where(n >= 5, area, np.nan)

    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        log_spread = np.log(spread(home_slots) + 1e-6) - np.log(spread(away_slots) + 1e-6)

    return {"bx": bx, "by": by, "ball_seen": ball_seen, "dh": dh, "da": da,
            "nearest": nearest, "mean_speed": centred_mean(mean_speed, C.SPEED_SMOOTH_FRAMES),
            "ball_speed": ball_speed, "log_spread": log_spread, "pitch": pitch}


def ball_feature_full(t):
    values = ["home_on", "home_near", "away_on", "away_near", "contest"] + GAP_VALUES
    r = C.CONTROL_RADIUS_M
    missing = t["nearest"] < 0
    dh, da = t["dh"], t["da"]
    with np.errstate(invalid="ignore"):
        tracked = np.select(
            [(dh < r) & (da < r), (dh <= da) & (dh < r), dh <= da, da < r],
            [values.index("contest"), values.index("home_on"), values.index("home_near"),
             values.index("away_on")],
            values.index("away_near"))
    codes = np.where(missing, 5 + gap_bin(missing), tracked).astype(np.int16)
    init_team = {"home_on": 0.22, "home_near": 0.22, "away_on": 0.05, "away_near": 0.12,
                 "contest": 0.05, "gap<1s": 0.06, "gap1-4s": 0.08, "gap4-10s": 0.10,
                 "gap>10s": 0.10}
    init_dead = {"home_on": 0.05, "home_near": 0.10, "away_on": 0.05, "away_near": 0.10,
                 "contest": 0.02, "gap<1s": 0.05, "gap1-4s": 0.10, "gap4-10s": 0.20,
                 "gap>10s": 0.33}
    return Feature("ball", values, codes, init_team, init_dead, learn_dead=False)


def _binned(x, edges, unknown_code):
    codes = np.searchsorted(np.asarray(edges), np.nan_to_num(x, nan=0.0), side="right")
    return np.where(np.isfinite(x), codes, unknown_code).astype(np.int16)


def pspeed_feature(t):
    values = ["<0.9", "0.9-1.3", "1.3-1.8", "1.8-2.5", ">2.5", "unknown"]
    codes = _binned(t["mean_speed"], C.PLAYER_SPEED_BINS_MS, 5)
    init_team = dict(zip(values, [0.08, 0.17, 0.30, 0.30, 0.15]))
    init_dead = dict(zip(values, [0.40, 0.30, 0.18, 0.09, 0.03]))
    return Feature("pspeed", values, codes, init_team, init_dead, team_bearing=False,
                   neutral=("unknown",))


def bspeed_feature(t):
    values = ["still", "rolling", "passed", "struck", "unknown"]
    codes = _binned(t["ball_speed"], C.BALL_SPEED_BINS_MS, 4)
    init_team = {"still": 0.10, "rolling": 0.45, "passed": 0.35, "struck": 0.10}
    init_dead = {"still": 0.40, "rolling": 0.40, "passed": 0.15, "struck": 0.05}
    return Feature("bspeed", values, codes, init_team, init_dead, team_bearing=False,
                   neutral=("unknown",))


def bpos_feature(t):
    """in / edge / out for a tracked ball; for a missing ball, whether it was last seen out."""
    length, width = t["pitch"]
    bx, by, seen = t["bx"], t["by"], t["ball_seen"]
    with np.errstate(invalid="ignore"):
        beyond = np.maximum.reduce([-bx, bx - length, -by, by - width])  # >0 means over a line
    out = seen & (beyond > C.OUT_MARGIN_M)
    edge = seen & ~out & (beyond > -C.EDGE_MARGIN_M)
    # Last tracked state carried through a gap.
    state = np.where(out, 2, np.where(seen, 0, -1))
    idx = np.where(state >= 0, np.arange(len(state)), 0)
    np.maximum.accumulate(idx, out=idx)
    last_out = (state[idx] == 2) & ~seen
    values = ["in", "edge", "out", "miss_after_out", "miss"]
    codes = np.select([out, edge, seen, last_out], [2, 1, 0, 3], 4).astype(np.int16)
    # "miss_after_out" (ball lost right after being seen over a line) is evidence; a plain
    # missing ball is not -- that is the ball feature's job.
    init_team = {"in": 0.90, "edge": 0.06, "out": 0.01, "miss_after_out": 0.03}
    init_dead = {"in": 0.55, "edge": 0.12, "out": 0.13, "miss_after_out": 0.20}
    return Feature("bpos", values, codes, init_team, init_dead, team_bearing=False,
                   neutral=("miss",))


def spread_feature(t):
    values = ["home_wider", "even", "away_wider", "unknown"]
    r = t["log_spread"]
    k = C.SPREAD_LOG_RATIO
    codes = np.select([~np.isfinite(r), r > k, r < -k], [3, 0, 2], 1).astype(np.int16)
    init_team = {"home_wider": 0.42, "even": 0.37, "away_wider": 0.21}
    init_dead = {"home_wider": 0.32, "even": 0.36, "away_wider": 0.32}
    return Feature("spread", values, codes, init_team, init_dead, learn_dead=False,
                   neutral=("unknown",))


# ------------------------------------------------------------------------------------------------
# camera feature (homography)
# ------------------------------------------------------------------------------------------------
def camera_centre(homography_path, n_frames, image_size=(1920, 1080), max_error=5.0):
    """Normalised pitch x of the point at the centre of the tactical image, NaN where unusable.

    The tactical camera follows play, so where it points is a coarse location of the ball that
    exists on every frame -- including the 41% where the ball itself is not tracked.
    """
    data = np.genfromtxt(homography_path, delimiter=",", names=True)
    if len(data) != n_frames:
        raise ValueError("homography: {} rows, expected {}".format(len(data), n_frames))
    names = ["h%d%d" % (i, j) for i in range(3) for j in range(3)]
    M = np.stack([data[n] for n in names], axis=1).reshape(-1, 3, 3)
    ok = (data["valid"] == 1) & np.isfinite(M).all(axis=(1, 2)) & \
        (data["reprojection_error"] <= max_error)
    x = np.full(n_frames, np.nan)
    centre = np.array([image_size[0] / 2.0, image_size[1] / 2.0, 1.0])
    with np.errstate(all="ignore"):
        inv = np.linalg.inv(np.where(ok[:, None, None], M, np.eye(3)))
        p = inv @ centre
        x = np.where(ok, p[:, 0] / p[:, 2], np.nan)
    return x


def camzone_feature(cam_x, home_attacks_right_by_frame):
    """Which team's attacking third the camera is looking at (team-relative), or unknown.

    home_attacks_right_by_frame: bool per frame, True where home attacks towards x = 1.
    """
    values = ["home_att", "mid", "away_att", "unknown"]
    x = np.where(home_attacks_right_by_frame, cam_x, 1.0 - cam_x)  # 1 = home's attacking end
    codes = np.select([~np.isfinite(x), x > 2.0 / 3.0, x < 1.0 / 3.0], [3, 0, 2], 1)
    init_team = {"home_att": 0.32, "mid": 0.47, "away_att": 0.21}
    init_dead = {"home_att": 0.32, "mid": 0.36, "away_att": 0.32}
    return Feature("camzone", values, codes.astype(np.int16), init_team, init_dead,
                   learn_team=False, learn_dead=False, neutral=("unknown",))
