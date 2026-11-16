"""Test on real tracker failures: the 20 SkillCorner broadcast-tracking matches (external).

    python scripts/test_skillcorner.py --configs "K I + restart kick" "M2 K, shape|ball + noball"

Each configuration of scripts/benchmark.py is trained on all ten Metrica + DFL matches, exactly as
the shipped model is, then applied unchanged to every SkillCorner match (possession/skillcorner.py),
which is never used for training, tuning or any earlier decision. SkillCorner labels name the team
in possession or none, so they test who has the ball, not dead ball. Reported per configuration:

  team agreement      on frames where SkillCorner names a team and the model says H or A, the
                      share where they agree
  called dead         share of frames where SkillCorner names a team that the model calls D
  spellF1, macroF1    the organisers' metrics with SkillCorner's unlabelled frames ungraded; with
                      no D in this ground truth they are a possession-only view

Needs external/skillcorner-opendata (git clone, MIT) and its tracking files in
external/skillcorner-tracking (scripts/download_skillcorner.py).
"""

import argparse
import json
import os
import pickle
import sys
import time
from multiprocessing import get_context

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "scripts"))

import benchmark as BM  # noqa: E402
from possession import config as C  # noqa: E402
from possession import skillcorner as SC  # noqa: E402
from possession import strict as ST  # noqa: E402

FITTED = {}
SC_ROOT = SC_TRACK = CACHE = None


def load(mid):
    path = os.path.join(CACHE, "SC-{}.pkl".format(mid))
    if os.path.exists(path):
        with open(path, "rb") as h:
            return pickle.load(h)
    g = SC.load_game(SC_ROOT, SC_TRACK, mid)
    with open(path, "wb") as h:
        pickle.dump(g, h, protocol=4)
    return g


def evaluate(mid):
    g = load(mid)
    ball = g["ball"]
    if BM.CORRECT:
        from possession.correct import correct_ball
        ball = correct_ball(ball, g["pitch"], g["blocks"])
    o = ST.make_obs(g["players"], ball, g["slot_team"], g["pitch"], g["blocks"],
                    labels=g["labels"])
    lab = g["labels"]
    rows = []
    for cfg, (bins, states, tables, trans, scale) in FITTED.items():
        f = BM.features(o, bins, BM.CONFIGS[cfg]["extras"], states)
        model = ST.FrozenHMM(f, tables, trans, scale, g["blocks"])
        path = model.decode(g["blocks"])
        pred = np.full(len(lab), 3, dtype=np.int8)
        pred[path >= 0] = path[path >= 0]
        team = (lab == 0) | (lab == 1)
        both = team & (pred <= 1)
        sf, mf = ST.spell_f1(lab, pred)
        rows.append({"config": cfg, "match": g["name"], "spell_f1": sf, "macro_f1": mf,
                     "team_agreement": float(np.mean(pred[both] == lab[both])),
                     "called_dead": float(np.mean(pred[team] == 2)),
                     "ball_missing": float(np.mean(~o["summary"]["ball_seen"][team]))})
    return rows


def main():
    global SC_ROOT, SC_TRACK, CACHE
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--configs", nargs="*", default=["K I + restart kick"])
    ap.add_argument("--skillcorner", default=os.path.join(ROOT, "external", "skillcorner-opendata"))
    ap.add_argument("--tracking", default=os.path.join(ROOT, "external", "skillcorner-tracking"))
    ap.add_argument("--cache", default=os.path.join(ROOT, "external", "cache"))
    ap.add_argument("--speed-chord", type=int, default=None)
    ap.add_argument("--correct", action="store_true")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    SC_ROOT, SC_TRACK, CACHE = a.skillcorner, a.tracking, a.cache
    if a.speed_chord is not None:
        C.SPEED_CHORD_FRAMES = a.speed_chord
    BM.CORRECT = a.correct
    t0 = time.time()
    BM.G = BM.prepare(a.cache, False, ())
    train = [g["train"] for g in BM.G]
    for cfg in a.configs:
        spec = BM.CONFIGS[cfg]
        pool = [g for g in BM.G if g["set"] in spec["train"]]
        FITTED[cfg] = BM.fit([g["train"] for g in pool], spec["extras"], [g["set"] for g in pool],
                             spec.get("trans_sets"), spec.get("scale_sets"))
        print("trained {} on {} external matches".format(cfg, len(train)), flush=True)
    rows = []
    with get_context("fork").Pool(a.workers) as pool:
        for rs in pool.imap_unordered(evaluate, SC.match_ids(SC_ROOT)):
            rows += rs
            print("  " + "   ".join("{}: spellF1 {:.3f} team {:.3f}".format(
                r["config"][:2], r["spell_f1"], r["team_agreement"]) for r in rs) +
                "   ({})".format(rs[0]["match"]), flush=True)
    print("\n{:<30} {:>8} {:>8} {:>10} {:>12}".format("config", "spellF1", "macroF1",
                                                      "team agr.", "called dead"))
    for cfg in a.configs:
        rs = [r for r in rows if r["config"] == cfg]
        print("{:<30} {:>8.4f} {:>8.4f} {:>10.4f} {:>12.4f}".format(
            cfg, *[np.mean([r[k] for r in rs]) for k in ("spell_f1", "macro_f1",
                                                         "team_agreement", "called_dead")]))
    if a.out:
        with open(a.out, "w") as h:
            json.dump(rows, h, indent=1)
    print("done in {:.0f} s".format(time.time() - t0))


if __name__ == "__main__":
    main()
