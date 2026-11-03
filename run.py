"""Regenerate the submission from the provided data.

    python run.py --team TEAM_NAME

writes submission/TEAM_NAME.csv (the possession file) and submission/TEAM_NAME_report.json (the
mode used, the learned emission tables and output statistics). Deterministic: same data, same
command, byte-identical file.
"""

import argparse
import os
import time

from possession.pipeline import run

ROOT = os.path.dirname(os.path.abspath(__file__))

if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    p.add_argument("--team", default="TEAM_NAME", help="names the output files (SUBMISSION.md)")
    p.add_argument("--mode", choices=("auto", "full", "basic"), default="auto",
                   help="full = raw tracking, basic = published nearest-player signal; "
                        "auto picks full when data/tactical/*_atd.txt exists")
    p.add_argument("--no-camera", action="store_true", help="ignore the homography")
    p.add_argument("--out-dir", default=os.path.join(ROOT, "submission"))
    p.add_argument("--quiet", action="store_true")
    a = p.parse_args()

    t0 = time.time()
    out = os.path.join(a.out_dir, "{}.csv".format(a.team))
    report = os.path.join(a.out_dir, "{}_report.json".format(a.team))
    run(mode=a.mode, out_path=out, use_camera=not a.no_camera, report_path=report,
        verbose=not a.quiet)
    print("done in {:.0f} s.  Validate:  python evaluation/validate_submission.py {}".format(
        time.time() - t0, os.path.relpath(out, ROOT)))
