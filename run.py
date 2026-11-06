"""Regenerate the submission from the provided data.

    python run.py

writes submission/YamineLamal.csv (the possession file) and submission/YamineLamal_report.json
(the model used, its frozen parameters' source and output statistics). Deterministic: same data,
same command, byte-identical file.

The submitted model is the strict-external one: every learned or tuned parameter comes from
other matches (models/external_params.json, trained by scripts/train_external.py on Metrica's
public sample games), and it is applied to the competition match frozen. --transductive runs the
earlier development model instead, which fits its parameters to the match's own tracking; it is
kept for reference only and is NOT the submission (see docs/TECHNICAL_REPORT.md, Section 7).
"""

import argparse
import os
import time

ROOT = os.path.dirname(os.path.abspath(__file__))

if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    p.add_argument("--team", default="YamineLamal", help="names the output files (SUBMISSION.md)")
    p.add_argument("--no-broadcast", action="store_true",
                   help="do not fuse the broadcast tracking (on by default when data/broadcast/ "
                        "has the broadcast ATD and homography)")
    p.add_argument("--out-dir", default=os.path.join(ROOT, "submission"))
    p.add_argument("--quiet", action="store_true")
    p.add_argument("--strict-external", action="store_true",
                   help="(the default; kept so older instructions still work)")
    p.add_argument("--transductive", action="store_true",
                   help="NOT the submission: the earlier model that fits its parameters to the "
                        "match's own tracking. For reference only")
    p.add_argument("--mode", choices=("auto", "full", "basic"), default="auto",
                   help="--transductive only: full = raw tracking, basic = published signal")
    p.add_argument("--no-camera", action="store_true", help="--transductive only")
    a = p.parse_args()

    t0 = time.time()
    out = os.path.join(a.out_dir, "{}.csv".format(a.team))
    report = os.path.join(a.out_dir, "{}_report.json".format(a.team))
    if a.transductive:
        from possession.pipeline import run
        print("NOTE: --transductive is the reference model, not the submission.")
        run(mode=a.mode, out_path=out, use_camera=not a.no_camera, report_path=report,
            verbose=not a.quiet, use_broadcast=not a.no_broadcast)
    else:
        from possession.pipeline import run_strict
        run_strict(out_path=out, report_path=report, verbose=not a.quiet,
                   use_broadcast=not a.no_broadcast)
    print("done in {:.0f} s.  Validate:  python evaluation/validate_submission.py {}".format(
        time.time() - t0, os.path.relpath(out, ROOT)))
