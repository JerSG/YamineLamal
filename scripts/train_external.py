"""Train the strict-external model on the public Metrica sample games (other matches).

    git clone https://github.com/metrica-sports/sample-data external/metrica-sample-data
    python scripts/train_external.py

Writes models/external_params.json, which run.py --strict-external applies, frozen, to the
competition match. Reads no competition data at all: it can run without data/.
"""

import argparse
import os
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from possession import external as E  # noqa: E402
from possession import strict as ST  # noqa: E402

if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--data", default=os.path.join(ROOT, "external", "metrica-sample-data",
                                                   "data"))
    ap.add_argument("--out", default=os.path.join(ROOT, "models", "external_params.json"))
    a = ap.parse_args()
    if not os.path.isdir(a.data):
        sys.exit("external data not found at {}\n  git clone https://github.com/metrica-sports/"
                 "sample-data external/metrica-sample-data".format(a.data))
    t0 = time.time()
    games = E.load_all(a.data)
    labels = [E.labels(g) for g in games]
    params = ST.train(games, labels)
    ST.save(params, a.out)
    print("transitions learned: mean possession {mean_alive_spell_s:.1f} s, {share:.0%} end in "
          "a stoppage, mean stoppage {mean_dead_spell_s:.1f} s".format(
              share=params["transition_stats"]["share_of_spells_ending_in_stoppage"],
              **params["transition_stats"]))
    print("wrote {} in {:.0f} s".format(os.path.relpath(a.out, ROOT), time.time() - t0))
