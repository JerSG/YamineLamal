"""Train the model on external matches only: Metrica's public sample games and the DFL open data.

    git clone https://github.com/metrica-sports/sample-data external/metrica-sample-data
    python scripts/download_dfl.py          # the DFL open data (CC BY 4.0) into external/dfl
    python scripts/train_external.py

Writes models/external_params.json, which run.py applies, frozen, to the competition match.
Reads no competition data at all: it can run without data/. Held-out evaluation of the same
training recipe is scripts/benchmark.py.

  Metrica sample games 1-3   labels derived mechanically from their event data
                             (possession/external.py), their own tracking gaps
  DFL open data, 7 matches   labels read from the tracking (BallStatus, BallPossession), the
                             way the competition's ground truth is made; ball gaps added by
                             the markov profile of possession/degrade.py (seed 0)
"""

import argparse
import os
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from possession import degrade as DG  # noqa: E402
from possession import dfl  # noqa: E402
from possession import external as E  # noqa: E402
from possession import strict as ST  # noqa: E402

SOURCE = ("Metrica Sports public sample games 1-3 (https://github.com/metrica-sports/sample-data), "
          "labels derived from their event data; DFL/IDSSE open data, 7 matches (Bassek et al. "
          "2025, Scientific Data 12, 195, doi:10.1038/s41597-025-04505-y; CC BY 4.0), labels "
          "from the tracking's ball status and possession, ball gaps added by the markov "
          "profile (seed 0)")

if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--metrica", default=os.path.join(ROOT, "external", "metrica-sample-data",
                                                      "data"))
    ap.add_argument("--dfl", default=os.path.join(ROOT, "external", "dfl"))
    ap.add_argument("--cache", default=os.path.join(ROOT, "external", "cache"))
    ap.add_argument("--out", default=os.path.join(ROOT, "models", "external_params.json"))
    a = ap.parse_args()
    if not os.path.isdir(a.metrica):
        sys.exit("Metrica sample data not found at {}\n  git clone https://github.com/"
                 "metrica-sports/sample-data external/metrica-sample-data".format(a.metrica))
    if not os.path.isdir(a.dfl) or not dfl.match_ids(a.dfl):
        sys.exit("DFL open data not found at {}\n  python scripts/download_dfl.py".format(a.dfl))
    t0 = time.time()
    met = E.load_all(a.metrica)
    mlab = [E.labels(g) for g in met]
    markov = dfl.missing_mask_model(met, mlab)
    games, labels = list(met), list(mlab)
    for g in dfl.load_all(a.dfl, a.cache):
        games.append(DG.apply_profile(g, "markov", 0, markov))
        labels.append(g["labels"])
    print("training on {} external games: {}".format(len(games), ", ".join(g["name"] for g in games)))
    params = ST.train(games, labels, SOURCE)
    ST.save(params, a.out)
    print("transitions learned: mean possession {mean_alive_spell_s:.1f} s, {share:.0%} end in "
          "a stoppage, mean stoppage {mean_dead_spell_s:.1f} s".format(
              share=params["transition_stats"]["share_of_spells_ending_in_stoppage"],
              **params["transition_stats"]))
    print("wrote {} in {:.0f} s".format(os.path.relpath(a.out, ROOT), time.time() - t0))
