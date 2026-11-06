"""Does an extra feature help? Leave-one-game-out on the external games only.

    python scripts/test_extensions.py

For each candidate set of features, and for each of the three external games: everything (bins,
extension thresholds, tables, transitions, scale) is learned on the other two games, then the
held-out game is decoded and scored with the organisers' spellF1. No competition data is read.
"""

import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from possession import config as C  # noqa: E402
from possession import extensions as X  # noqa: E402
from possession import external as E  # noqa: E402
from possession import strict as ST  # noqa: E402


def build(summaries, quants, bins, thresholds, extra):
    """Per game: base tracking features + the selected extension features."""
    out = []
    with ST.use_bins(bins):
        for t, q in zip(summaries, quants):
            feats = ST.tracking_features(None, None, None, None, summary=t)
            if extra:
                feats += [f for f in X.features(q, thresholds) if f.name in extra]
            out.append(feats)
    return out


def loo(games, labels, summaries, quants, extra):
    scores = []
    for i in range(len(games)):
        tr = [j for j in range(len(games)) if j != i]
        bins = ST.derive_bins([summaries[j] for j in tr], [labels[j] for j in tr])
        thr = X.derive_thresholds([quants[j] for j in tr])
        feats = build(summaries, quants, bins, thr, extra)
        tables = ST.fit_tables([feats[j] for j in tr], [labels[j] for j in tr])
        trans, _ = ST.fit_transitions([labels[j] for j in tr])
        scale, _ = ST.tune_scale([feats[j] for j in tr], [labels[j] for j in tr], tables, trans)
        sf, mf = ST.spell_f1(labels[i], ST.decode_game(feats[i], tables, trans, scale, labels[i]))
        scores.append((sf, mf, scale))
    return scores


def main():
    data = os.path.join(ROOT, "external", "metrica-sample-data", "data")
    games = E.load_all(data)
    labels = [E.labels(g) for g in games]
    pitch = C.STRICT_EXTERNAL_PITCH
    summaries = [ST.summarise(g["players"], g["ball"], g["slot_team"], pitch) for g in games]
    quants = [X.quantities(g["players"], g["ball"], g["slot_team"], pitch,
                           ST.blocks_from_labels(lab), t)
              for g, lab, t in zip(games, labels, summaries)]
    candidates = [("base (current model)", ())] + [("+ " + n, (n,)) for n in X.NAMES] + \
        [("+ all four", X.NAMES)]
    print("{:<24} {:>7} {:>7} {:>7} {:>8}   (held-out spellF1 per game; mean)".format(
        "features", "game1", "game2", "game3", "mean"))
    base = None
    for name, extra in candidates:
        s = loo(games, labels, summaries, quants, extra)
        sf = [x[0] for x in s]
        mean = float(np.mean(sf))
        if base is None:
            base = sf
        better = sum(a > b for a, b in zip(sf, base))
        print("{:<24} {:>7.4f} {:>7.4f} {:>7.4f} {:>8.4f}   better on {}/3 games{}".format(
            name, *sf, mean, better, "" if extra else " (reference)"))
        sys.stdout.flush()


if __name__ == "__main__":
    main()
