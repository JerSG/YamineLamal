"""Candidate extra features for the strict-external model, each kept only if it improves the
held-out external games (scripts/test_extensions.py).

All of them are computed from tracking alone, identically on the external games and on the
competition match, and their thresholds are learned on the external games with the same
equal-share rule as the other bins. Attack directions are not read from metadata: in each graded
block, the team whose players stand on average at lower x defends the low-x goal.

  pressure    which team has more players within PRESS_RADIUS_M of the ball
  territory   which team's attacking third the ball is in
  advance     which team's centroid moves faster towards the opponent's goal
  carrier     who has been on the ball, and for how long (short touch or sustained control)
"""

import numpy as np

from . import config as C
from . import features as F
from .sources import HOME, AWAY

PRESS_RADIUS_M = 5.0        # [football] "near the ball": a pressing / supporting distance
ADVANCE_SMOOTH_FRAMES = 12  # [model] +/- 0.5 s smoothing of the centroid velocity
NAMES = ("pressure", "territory", "advance", "carrier")


def _home_attacks_right(px, slot_team, blocks, n):
    """Per frame: True where home attacks towards x = 1, inferred per block from positions."""
    right = np.zeros(n, dtype=bool)
    home, away = slot_team == HOME, slot_team == AWAY
    for s, e in blocks:
        with np.errstate(all="ignore"):
            import warnings
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", RuntimeWarning)
                h = np.nanmean(px[s:e][:, home])
                a = np.nanmean(px[s:e][:, away])
        right[s:e] = bool(h < a)  # home stands deeper on the low-x side: it attacks +x
    return right


def quantities(players, ball, slot_team, pitch, blocks, summary):
    """Raw per-frame quantities the four features are binned from."""
    length, width = pitch
    n = len(ball)
    px, py = players[..., 0] * length, players[..., 1] * width
    seen = np.isfinite(px) & np.isfinite(py)
    bx, by = ball[:, 0] * length, ball[:, 1] * width
    ball_seen = np.isfinite(bx) & np.isfinite(by)
    home, away = slot_team == HOME, slot_team == AWAY
    right = _home_attacks_right(px, slot_team, blocks, n)

    # pressure: players within PRESS_RADIUS_M of the ball, home minus away
    d = np.sqrt((px - bx[:, None]) ** 2 + (py - by[:, None]) ** 2)
    near = seen & (d <= PRESS_RADIUS_M)
    press = near[:, home].sum(axis=1).astype(float) - near[:, away].sum(axis=1)
    press[~ball_seen] = np.nan

    # territory: ball x in the home team's attacking coordinates (0 own goal .. 1 opponent's)
    terr = np.where(right, bx / length, 1.0 - bx / length)
    terr[~ball_seen] = np.nan

    # advance: forward speed of each team's centroid, home minus away
    def centroid_x(mask):
        x = np.where(seen[:, mask], px[:, mask], 0.0)
        k = seen[:, mask].sum(axis=1)
        with np.errstate(invalid="ignore", divide="ignore"):
            return np.where(k >= 5, x.sum(axis=1) / np.maximum(k, 1), np.nan)
    ch, ca = centroid_x(home), centroid_x(away)
    vh = np.full(n, np.nan)
    va = np.full(n, np.nan)
    vh[1:] = np.diff(ch) * C.FPS
    va[1:] = np.diff(ca) * C.FPS
    sign = np.where(right, 1.0, -1.0)  # home forward = +x when home attacks right
    adv = F.centred_mean(sign * vh, ADVANCE_SMOOTH_FRAMES) - \
        F.centred_mean(-sign * va, ADVANCE_SMOOTH_FRAMES)
    for s, e in blocks:  # no velocity across block edges
        adv[s] = np.nan

    # carrier: signed length of the current run of one team on the ball (within 2 m)
    r = C.CONTROL_RADIUS_M
    on = np.where(summary["nearest"] < 0, -1,
                  np.where((summary["dh"] < r) & ~(summary["da"] < r), HOME,
                           np.where((summary["da"] < r) & ~(summary["dh"] < r), AWAY, -1)))
    runs = F.run_lengths(on)
    carrier = np.where(on == HOME, runs, np.where(on == AWAY, -runs, 0)).astype(float)
    return {"press": press, "terr": terr, "adv": adv, "carrier": carrier}


def derive_thresholds(qs):
    """Equal-share thresholds from the (external) training games' quantities."""
    def pct(key, level, transform=np.abs, positive=False):
        x = np.concatenate([q[key] for q in qs])
        x = x[np.isfinite(x)]
        if positive:
            x = x[x != 0]
        return float(np.percentile(transform(x), level)) if len(x) else 1.0
    return {
        "advance": round(pct("adv", 100.0 / 3), 4),                  # |diff| below: "even"
        "carrier": round(pct("carrier", 50, positive=True), 1),      # median control run
    }


def features(q, thresholds):
    """The four candidate features (team-bearing, mirrored like the others)."""
    out = []
    press = q["press"]
    codes = np.select([~np.isfinite(press), press > 0, press < 0], [3, 0, 2], 1)
    out.append(F.Feature("pressure", ["home_more", "even", "away_more", "unknown"],
                         codes.astype(np.int16),
                         {"home_more": 0.45, "even": 0.30, "away_more": 0.25},
                         {"home_more": 0.33, "even": 0.34, "away_more": 0.33},
                         neutral=("unknown",)))
    terr = q["terr"]
    codes = np.select([~np.isfinite(terr), terr > 2.0 / 3, terr < 1.0 / 3], [3, 0, 2], 1)
    out.append(F.Feature("territory", ["home_att", "mid", "away_att", "unknown"],
                         codes.astype(np.int16),
                         {"home_att": 0.32, "mid": 0.46, "away_att": 0.22},
                         {"home_att": 0.32, "mid": 0.36, "away_att": 0.32},
                         neutral=("unknown",)))
    adv, k = q["adv"], thresholds["advance"]
    codes = np.select([~np.isfinite(adv), adv > k, adv < -k], [3, 0, 2], 1)
    out.append(F.Feature("advance", ["home_fwd", "even", "away_fwd", "unknown"],
                         codes.astype(np.int16),
                         {"home_fwd": 0.40, "even": 0.35, "away_fwd": 0.25},
                         {"home_fwd": 0.33, "even": 0.34, "away_fwd": 0.33},
                         neutral=("unknown",)))
    c, k = q["carrier"], thresholds["carrier"]
    codes = np.select([c >= k, c > 0, c <= -k, c < 0], [0, 1, 3, 2], 4)
    out.append(F.Feature("carrier", ["home_long", "home_short", "away_short", "away_long",
                                     "none"], codes.astype(np.int16),
                         {"home_long": 0.35, "home_short": 0.20, "away_short": 0.10,
                          "away_long": 0.05, "none": 0.30},
                         {"home_long": 0.05, "home_short": 0.08, "away_short": 0.08,
                          "away_long": 0.05, "none": 0.74}))
    return out
