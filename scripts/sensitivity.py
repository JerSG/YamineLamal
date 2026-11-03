"""How much does the output move when each prior moves? Unsupervised robustness check.

    python scripts/sensitivity.py

Re-runs the pipeline with one setting changed at a time and reports, against the reference run:
frame agreement on graded frames, and the number of possessions. Reads no ground truth -- it
compares the pipeline only with itself.

The broadcast offset search runs once (it does not depend on any setting varied here); the
acceptance thresholds are then re-applied per variant, exactly as broadcast.align applies them.
About 15-20 minutes in total.
"""

import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import games  # noqa: E402
from possession import broadcast as B  # noqa: E402
from possession import config as C  # noqa: E402
from possession import pipeline as P  # noqa: E402
from possession import sources as S  # noqa: E402

VARIANTS = [
    ("reference", {}),
    ("alive spell 10 s", {"MEAN_ALIVE_SPELL_S": 10.0}),
    ("alive spell 15 s", {"MEAN_ALIVE_SPELL_S": 15.0}),
    ("stoppage share 0.25", {"P_SPELL_ENDS_IN_STOPPAGE": 0.25}),
    ("stoppage share 0.40", {"P_SPELL_ENDS_IN_STOPPAGE": 0.40}),
    ("dead spell 12 s", {"MEAN_DEAD_SPELL_S": 12.0}),
    ("dead spell 25 s", {"MEAN_DEAD_SPELL_S": 25.0}),
    ("prior strength 500", {"PRIOR_STRENGTH": 500.0}),
    ("prior strength 8000", {"PRIOR_STRENGTH": 8000.0}),
    ("control radius 1.5 m", {"CONTROL_RADIUS_M": 1.5}),
    ("control radius 3 m", {"CONTROL_RADIUS_M": 3.0}),
    ("no camera", {"_no_camera": True}),
    ("broadcast: strict fit 1.0 m", {"BC_MAX_COST_M": 1.0}),
    ("broadcast: loose fit 2.0 m", {"BC_MAX_COST_M": 2.0}),
    ("broadcast: ball check 1.5 m", {"BC_MAX_BALL_ERR_M": 1.5}),
    ("broadcast: ball check 3.0 m", {"BC_MAX_BALL_ERR_M": 3.0}),
    ("no broadcast", {"_no_broadcast": True}),
]

ACCEPTANCE = ("BC_MAX_COST_M", "BC_MIN_MARGIN_M", "BC_MIN_BALL_CHECK_FRAMES", "BC_MAX_BALL_ERR_M")


def broadcast_search():
    """Run the broadcast offset search once, with acceptance switched fully open, and return
    what is needed to re-apply any acceptance thresholds later."""
    paths = games.asset_paths()
    bpaths = games.asset_paths(source="broadcast")
    tm = S.read_atd_meta(paths["atd_meta"])
    bm = S.read_atd_meta(bpaths["atd_meta"])
    tp, tb = S.read_atd(paths["atd_raw"], len(tm["slot_team"]), games.source_frames("tactical"))
    n_bc = games.source_frames("broadcast")
    bp, bb = S.read_atd(bpaths["atd_raw"], len(bm["slot_team"]), n_bc)
    aim, ok = B.camera_aim(bpaths["homography"], n_bc)
    saved = {k: getattr(C, k) for k in ACCEPTANCE}
    C.BC_MAX_COST_M, C.BC_MIN_MARGIN_M = float("inf"), -float("inf")
    C.BC_MIN_BALL_CHECK_FRAMES, C.BC_MAX_BALL_ERR_M = 10 ** 9, float("inf")
    try:
        _, table = B.align(bp, bb, bm["slot_team"], tp, tb, tm["slot_team"], aim, ok,
                           games.GAMES[games.GAME]["pitch"], verbose=False)
    finally:
        for k, v in saved.items():
            setattr(C, k, v)
    return table, bb, n_bc


def mapping(table, n_bc, n_tac):
    """broadcast.align's acceptance rule, re-applied with the current config values."""
    tac_of_bc = np.full(n_bc, -1, dtype=np.int64)
    for r in table:
        accepted = (r["cost_m"] <= C.BC_MAX_COST_M and
                    r["runner_up_m"] - r["cost_m"] >= C.BC_MIN_MARGIN_M)
        if accepted and r["ball_check_frames"] >= C.BC_MIN_BALL_CHECK_FRAMES:
            accepted = r["ball_median_err_m"] <= C.BC_MAX_BALL_ERR_M
        if accepted:
            t = np.arange(r["start"], r["end"]) + r["offset"]
            inside = (t >= 0) & (t < n_tac)
            tac_of_bc[r["start"]:r["end"]][inside] = t[inside]
    return tac_of_bc


def main():
    blocks = [(s, e) for _, s, e in P.graded_blocks()]
    graded = np.concatenate([np.arange(s, e) for s, e in blocks])
    n_tac = games.source_frames("tactical")
    print("broadcast offset search (once) ...")
    sys.stdout.flush()
    table, bc_ball, n_bc = broadcast_search()

    reference = None
    print("{:<30} {:>9} {:>11} {:>10}".format("variant", "agree", "possessions", "mean s"))
    for name, overrides in VARIANTS:
        saved = {k: getattr(C, k) for k in overrides if not k.startswith("_")}
        for k, v in overrides.items():
            if not k.startswith("_"):
                setattr(C, k, v)
        try:
            hook = None
            if not overrides.get("_no_broadcast"):
                tac_of_bc = mapping(table, n_bc, n_tac)

                def hook(players, ball, meta, tac_of_bc=tac_of_bc):
                    filled, mask = B.fill_ball(ball, bc_ball, tac_of_bc)
                    return filled, {"filled": int(mask.sum())}
            labels, info = P.run(use_camera=not overrides.get("_no_camera"), verbose=False,
                                 ball_hook=hook, use_broadcast=False)
        finally:
            for k, v in saved.items():
                setattr(C, k, v)
        if reference is None:
            reference = labels
        agree = float(np.mean(labels[graded] == reference[graded]))
        print("{:<30} {:>9.4f} {:>11} {:>10.1f}".format(
            name, agree, info["possessions"], info["mean_possession_s"]))
        sys.stdout.flush()


if __name__ == "__main__":
    main()
