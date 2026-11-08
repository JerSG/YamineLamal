"""Regenerate the submission from the provided data.

    python run.py

writes submission/YamineLamal.csv (the possession file) and submission/YamineLamal_report.json
(the model used, its frozen parameters' source and output statistics). Deterministic: same data,
same command, byte-identical file.

Then it writes submission/viewer.html, a local replay of the tracking with those labels, and
opens it in the browser (--no-viewer to skip, --no-open to write it without opening). The page
is built after the possession file is written and does not change it.

Every learned or tuned parameter comes from other matches (models/external_params.json, trained by
scripts/train_external.py on Metrica's public sample games); the model is applied to the
competition match frozen.
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
    p.add_argument("--no-viewer", action="store_true",
                   help="do not write the replay page submission/viewer.html")
    p.add_argument("--no-open", action="store_true",
                   help="write the replay page but do not open the browser")
    a = p.parse_args()

    from possession.pipeline import run_strict
    t0 = time.time()
    out = os.path.join(a.out_dir, "{}.csv".format(a.team))
    report = os.path.join(a.out_dir, "{}_report.json".format(a.team))
    labels, _, inputs = run_strict(out_path=out, report_path=report, verbose=not a.quiet,
                                   use_broadcast=not a.no_broadcast, return_inputs=True)
    print("done in {:.0f} s.  Validate:  python evaluation/validate_submission.py {}".format(
        time.time() - t0, os.path.relpath(out, ROOT)))
    if not a.no_viewer:
        # after the submission is written: a failure here never affects the possession file
        try:
            import webbrowser
            from possession.viewer import write_match_page
            page = write_match_page(os.path.join(a.out_dir, "viewer.html"), inputs["players"],
                                    inputs["ball"], inputs["ball_filled"], inputs["meta"],
                                    labels, a.team)
            print("replay page: {} (local only, do not publish)".format(
                os.path.relpath(page, ROOT)))
            if not a.no_open:
                webbrowser.open("file://" + os.path.abspath(page))
        except Exception as error:      # noqa: BLE001
            print("replay page skipped: {}".format(error))
