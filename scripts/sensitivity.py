"""How much does the output move when each prior moves? Unsupervised robustness check.

    python scripts/sensitivity.py

Re-runs the pipeline with one config value changed at a time and reports, against the reference
run: frame agreement on graded frames, and the number of possessions. Reads no ground truth -- it
compares the pipeline only with itself. Takes ~40 minutes (12 full runs).
"""

import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from possession import config as C  # noqa: E402
from possession import pipeline as P  # noqa: E402

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
]


def main():
    blocks = [(s, e) for _, s, e in P.graded_blocks()]
    graded = np.concatenate([np.arange(s, e) for s, e in blocks])
    reference = None
    print("{:<24} {:>9} {:>11} {:>10}".format("variant", "agree", "possessions", "mean s"))
    for name, overrides in VARIANTS:
        saved = {k: getattr(C, k) for k in overrides if not k.startswith("_")}
        for k, v in overrides.items():
            if not k.startswith("_"):
                setattr(C, k, v)
        try:
            labels, info = P.run(use_camera=not overrides.get("_no_camera"), verbose=False)
        finally:
            for k, v in saved.items():
                setattr(C, k, v)
        if reference is None:
            reference = labels
        agree = float(np.mean(labels[graded] == reference[graded]))
        print("{:<24} {:>9.4f} {:>11} {:>10.1f}".format(
            name, agree, info["possessions"], info["mean_possession_s"]))
        sys.stdout.flush()


if __name__ == "__main__":
    main()
