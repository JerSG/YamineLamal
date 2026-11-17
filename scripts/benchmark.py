"""Benchmark candidate models on external matches only (Metrica sample games + DFL open data).

    python scripts/benchmark.py                       # development: leave one game out
    python scripts/benchmark.py --final               # once, at the end: the reserved matches

Development: for every development game, everything (bins, extra models, emission tables,
transitions, evidence scale) is learned on the other development games of the configuration's
training sets, then the held-out game is decoded and scored with the organisers' spellF1.
Metrica games are scored once, with their own tracking gaps. DFL games are scored under every
gap profile of possession/degrade.py and three seeds. Two DFL matches are reserved: they are
neither trained on nor scored until --final, which trains on all development games and scores
the reserved ones.

The decision rule is fixed before running (docs/TECHNICAL_REPORT.md): a change is adopted when
its mean spellF1 on the DFL development matches is higher (DFL labels are made the way the
competition's ground truth is: ball status and possession from the tracking), its Metrica mean
is not more than 0.03 lower, and --final does not reverse the gain. No competition data is read.

Needs external/metrica-sample-data and external/dfl (see README).
"""

import argparse
import json
import os
import sys
import time
from multiprocessing import get_context

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from possession import config as C  # noqa: E402
from possession import degrade as DG  # noqa: E402
from possession import dfl  # noqa: E402
from possession import external as E  # noqa: E402
from possession import strict as ST  # noqa: E402

PITCH = C.STRICT_EXTERNAL_PITCH
RESERVED = ("DFL-MAT-J03WQQ", "DFL-MAT-J03WR9")      # chosen by a fixed rule: last two by id
SEEDS = (1, 2, 3)
TRAIN_PROFILE, TRAIN_SEED = "markov", 0

# configuration name -> training sets and extra features
CONFIGS = {
    "A metrica": {"train": ("metrica",), "extras": ()},
    "B metrica+dfl": {"train": ("metrica", "dfl"), "extras": ()},
    "C B + touch": {"train": ("metrica", "dfl"), "extras": ("touch",)},
    "D B + lasttouch": {"train": ("metrica", "dfl"), "extras": ("lasttouch",)},
    "E B + shape": {"train": ("metrica", "dfl"), "extras": ("shape",)},
    "F B + lasttouch + shape": {"train": ("metrica", "dfl"), "extras": ("lasttouch", "shape")},
    "G B + all three": {"train": ("metrica", "dfl"), "extras": ("touch", "lasttouch", "shape")},
    "H F + gapctx": {"train": ("metrica", "dfl"), "extras": ("lasttouch", "shape", "gapctx")},
    "I F + still": {"train": ("metrica", "dfl"), "extras": ("lasttouch", "shape", "still")},
    "J F + gapctx + still": {"train": ("metrica", "dfl"),
                             "extras": ("lasttouch", "shape", "gapctx", "still")},
    "K I + restart kick": {"train": ("metrica", "dfl"),
                           "extras": ("lasttouch", "shape", "still", "restart")},
    "S K + slowdown": {"train": ("metrica", "dfl"),
                       "extras": ("lasttouch", "shape", "still", "restart", "slowdown")},
    "N K, DFL transitions + scale": {"train": ("metrica", "dfl"),
                                     "extras": ("lasttouch", "shape", "still", "restart"),
                                     "trans_sets": ("dfl",), "scale_sets": ("dfl",)},
    "N2 K, DFL transitions": {"train": ("metrica", "dfl"),
                              "extras": ("lasttouch", "shape", "still", "restart"),
                              "trans_sets": ("dfl",)},
    "M1 K + noball": {"train": ("metrica", "dfl"),
                      "extras": ("lasttouch", "shape", "still", "restart", "noball")},
    "M2 K, shape|ball + noball": {"train": ("metrica", "dfl"),
                                  "extras": ("lasttouch", "shape_seen", "still", "restart",
                                             "noball")},
}


class Extra(object):
    """A candidate feature: fit(train observations) -> state; features(obs, state) -> [Feature]."""

    def __init__(self, fit, features):
        self.fit, self.features = fit, features


def _extras():
    from possession import deadball, shape, touch
    return {"gapctx": Extra(lambda train: None,
                            lambda o, st: [deadball.gapctx_feature(o["summary"], o["blocks"])]),
            "still": Extra(lambda train: None, lambda o, st: [deadball.still_feature(o["summary"])]),
            "slowdown": Extra(lambda train: deadball.fit_slowdown(train),
                              lambda o, st: [deadball.slowdown_feature(o["summary"], st)]),
            "restart": Extra(lambda train: None,
                             lambda o, st: [deadball.restart_feature(o["summary"])]),
            "touch": Extra(touch.fit, touch.touch_feature),
            "lasttouch": Extra(touch.fit, touch.lasttouch_feature),
            "shape": Extra(shape.fit, shape.shape_feature),
            "shape_seen": Extra(shape.fit, shape.shape_seen_feature),
            "noball": Extra(shape.fit_noball, shape.noball_feature)}


EXTRAS = _extras()

G = None           # filled by prepare(): shared with the worker processes (fork)


def load_dfl(cache_dir):
    return dfl.load_all(os.path.join(ROOT, "external", "dfl"), cache_dir)


CORRECT = False          # --correct: repair the ball track (possession/correct.py) first


def obs_entry(game, labels):
    if CORRECT:
        from possession.correct import correct_ball
        game = dict(game, ball=correct_ball(game["ball"], PITCH, ST.blocks_from_labels(labels)))
    s = ST.summarise(game["players"], game["ball"], game["slot_team"], PITCH)
    return {"game": game, "labels": labels, "summary": s,
            "blocks": ST.blocks_from_labels(labels)}


def prepare(cache_dir, final, reserved_names=RESERVED):
    met = E.load_all(os.path.join(ROOT, "external", "metrica-sample-data", "data"))
    mlab = [E.labels(g) for g in met]
    markov = dfl.missing_mask_model(met, mlab)
    games = []
    for g, lab in zip(met, mlab):
        games.append({"name": g["name"], "set": "metrica", "reserved": False,
                      "train": obs_entry(g, lab), "test": {("own", 0): None}})
        games[-1]["test"][("own", 0)] = games[-1]["train"]
    for g in load_dfl(cache_dir):
        reserved = g["name"] in reserved_names
        entry = {"name": g["name"], "set": "dfl", "reserved": reserved,
                 "train": obs_entry(DG.apply_profile(g, TRAIN_PROFILE, TRAIN_SEED, markov), g["labels"]),
                 "test": {}}
        if reserved == final:
            for prof in DG.PROFILES:
                for seed in SEEDS:
                    entry["test"][(prof, seed)] = obs_entry(
                        DG.apply_profile(g, prof, seed, markov), g["labels"])
        games.append(entry)
    return games


def features(o, bins, extras, states):
    with ST.use_bins(bins):
        f = ST.tracking_features(None, None, None, None, summary=o["summary"])
    for name in extras:
        f += EXTRAS[name].features(o, states[name])
    return f


def fit(train, extras, sets=None, trans_sets=None, scale_sets=None):
    """sets: the dataset of each training observation; trans_sets / scale_sets: learn the
    transitions / tune the scale on those datasets only (default: all training games)."""
    bins = ST.derive_bins([o["summary"] for o in train], [o["labels"] for o in train])
    states = {n: EXTRAS[n].fit(train) for n in extras}
    feats = [features(o, bins, extras, states) for o in train]
    labs = [o["labels"] for o in train]
    tables = ST.fit_tables(feats, labs)
    pick = lambda which: [k for k in range(len(train))                       # noqa: E731
                          if which is None or sets is None or sets[k] in which]
    tk, sk = pick(trans_sets), pick(scale_sets)
    trans, _ = ST.fit_transitions([labs[k] for k in tk])
    scale, _ = ST.tune_scale([feats[k] for k in sk], [labs[k] for k in sk], tables, trans)
    return bins, states, tables, trans, scale


def run_job(job):
    cfg, i = job
    spec = CONFIGS[cfg]
    test_game = G[i]
    if test_game["reserved"]:          # final: train on every development game
        pool = [g for g in G if not g["reserved"] and g["set"] in spec["train"]]
    else:
        pool = [g for k, g in enumerate(G) if k != i and not g["reserved"]
                and g["set"] in spec["train"]]
    bins, states, tables, trans, scale = fit([g["train"] for g in pool], spec["extras"],
                                             [g["set"] for g in pool], spec.get("trans_sets"),
                                             spec.get("scale_sets"))
    rows = []
    for (prof, seed), o in sorted(test_game["test"].items()):
        f = features(o, bins, spec["extras"], states)
        pred = ST.decode_game(f, tables, trans, scale, o["labels"])
        sf, mf = ST.spell_f1(o["labels"], pred)
        rows.append({"config": cfg, "game": test_game["name"], "set": test_game["set"],
                     "profile": prof, "seed": seed, "spell_f1": sf, "macro_f1": mf,
                     "scale": scale})
    return rows


def summarise_rows(rows, reference):
    by = {}
    for r in rows:
        by.setdefault(r["config"], []).append(r)
    print("\n{:<28} {:>9} {:>9}  {}".format("config", "Metrica", "DFL",
                                          "DFL per profile (" + "/".join(DG.PROFILES) + ")"))
    ref_game = {}
    for cfg in CONFIGS:
        rs = by.get(cfg, [])
        if not rs:
            continue
        met = [r["spell_f1"] for r in rs if r["set"] == "metrica"]
        dl = [r["spell_f1"] for r in rs if r["set"] == "dfl"]
        prof = [np.mean([r["spell_f1"] for r in rs if r["set"] == "dfl" and r["profile"] == p])
                if any(r["profile"] == p for r in rs) else float("nan") for p in DG.PROFILES]
        game_mean = {}
        for r in rs:
            if r["set"] == "dfl":
                game_mean.setdefault(r["game"], []).append(r["spell_f1"])
        game_mean = {k: float(np.mean(v)) for k, v in game_mean.items()}
        if cfg == reference:
            ref_game = game_mean
        wins = sum(game_mean[k] > ref_game.get(k, np.inf) for k in game_mean) if ref_game else 0
        print("{:<28} {:>9} {:>9.4f}  {}   {}".format(
            cfg, "{:.4f}".format(np.mean(met)) if met else "-", np.mean(dl),
            " / ".join("{:.4f}".format(x) for x in prof),
            "(reference)" if cfg == reference else "better on {}/{} DFL games".format(
                wins, len(game_mean))))


def main():
    global G
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--final", action="store_true", help="score the reserved DFL matches")
    ap.add_argument("--configs", nargs="*", default=list(CONFIGS))
    ap.add_argument("--cache", default=os.path.join(ROOT, "external", "cache"),
                    help="where parsed DFL matches are cached (gitignored)")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--correct", action="store_true",
                    help="repair the ball track before the features (possession/correct.py)")
    ap.add_argument("--speed-chord", type=int, default=None,
                    help="override config.SPEED_CHORD_FRAMES for every configuration")
    ap.add_argument("--reserved", nargs="*", default=list(RESERVED),
                    help="DFL matches held back (default: the two fixed in RESERVED)")
    ap.add_argument("--out", default=None, help="write every score to this JSON file")
    a = ap.parse_args()
    if a.speed_chord is not None:
        C.SPEED_CHORD_FRAMES = a.speed_chord
    global CORRECT
    CORRECT = a.correct
    os.makedirs(a.cache, exist_ok=True)
    t0 = time.time()
    G = prepare(a.cache, a.final, tuple(a.reserved))
    print("prepared {} games in {:.0f} s".format(len(G), time.time() - t0))
    jobs = [(c, i) for c in a.configs for i, g in enumerate(G)
            if g["test"] and (g["reserved"] == a.final)]
    rows = []
    with get_context("fork").Pool(a.workers) as pool:
        for rs in pool.imap_unordered(run_job, jobs):
            rows += rs
            r = rs[0]
            print("  {:<28} {:<16} mean spellF1 {:.4f} over {} runs".format(
                r["config"], r["game"], np.mean([x["spell_f1"] for x in rs]), len(rs)),
                flush=True)
    summarise_rows(rows, a.configs[0])
    if a.out:
        with open(a.out, "w") as h:
            json.dump(rows, h, indent=1)
    print("done in {:.0f} s".format(time.time() - t0))


if __name__ == "__main__":
    main()
