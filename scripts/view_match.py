"""Replay the tracking with the possession labels in your browser (one self-contained HTML file).

    python run.py                          # writes the submission AND this page (see run.py)
    python scripts/view_match.py           # rebuild the page from submission/YamineLamal.csv
    python scripts/view_match.py --external 1   # a Metrica sample game, no match data needed

The competition view shows the tracking the model read (with the ball positions filled from the
broadcast drawn hollow) and the labels in our submission file. No ground truth is read or shown.
The page embeds tracking, so it is written to submission/ (gitignored) and must not be published.

The --external view shows a public Metrica sample game with two timelines: the frozen model's
labels and the labels derived from that game's events. The shipped parameters were trained on all
three sample games, so this view is in-sample and only illustrates what the model does.
"""

import argparse
import os
import sys
import webbrowser

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import games  # noqa: E402
from possession import config as C  # noqa: E402
from possession import sources as S  # noqa: E402
from possession import viewer as V  # noqa: E402
from possession.hmm import STATE_LABELS  # noqa: E402


def read_labels(path, n_frames):
    labels = np.full(n_frames, "X", dtype="<U1")
    with open(path) as handle:
        next(handle)
        for line in handle:
            frame, label = line.strip().split(",")
            labels[int(frame)] = label
    return labels


def competition(a):
    csv = os.path.join(ROOT, "submission", "{}.csv".format(a.team))
    if not os.path.exists(csv):
        sys.exit("{} not found: run  python run.py  first".format(os.path.relpath(csv, ROOT)))
    paths = games.asset_paths()
    n_frames = games.source_frames("tactical")
    meta = S.read_atd_meta(paths["atd_meta"])
    print("reading tracking ...")
    players, ball = S.read_atd(paths["atd_raw"], len(meta["slot_team"]), n_frames)
    filled = None
    if not a.no_broadcast:
        from possession.pipeline import broadcast_hook
        hook = broadcast_hook(verbose=False)
        if hook is not None:
            print("aligning broadcast shots (as run.py does) ...")
            before = np.isfinite(ball[:, 0])
            ball, _ = hook(players, ball, meta)
            filled = np.isfinite(ball[:, 0]) & ~before
    labels = read_labels(csv, n_frames)
    out = a.out or os.path.join(ROOT, "submission", "viewer.html")
    return V.write_match_page(out, players, ball, filled, meta, labels, a.team, a.step)


def external(a):
    from possession import external as E
    from possession import strict as ST
    from possession.pipeline import STRICT_PARAMS
    data = os.path.join(ROOT, "external", "metrica-sample-data", "data")
    if not os.path.isdir(data):
        sys.exit("external data not found: git clone https://github.com/metrica-sports/"
                 "sample-data external/metrica-sample-data")
    g = (E.load_epts_game(data, 3) if a.external == 3
         else E.load_csv_game(data, a.external))
    truth = E.labels(g)
    blocks = ST.blocks_from_labels(truth)
    params = ST.load(STRICT_PARAMS)
    path, _, _ = ST.apply(params, g["players"], g["ball"], g["slot_team"], [], len(truth), blocks,
                          C.STRICT_EXTERNAL_PITCH)
    pred = np.full(len(truth), "X", dtype="<U1")
    pred[path >= 0] = STATE_LABELS[path[path >= 0]]
    letters = np.array(["H", "A", "D", "X"])
    out = a.out or os.path.join(ROOT, "submission", "viewer_sample_game_{}.html".format(a.external))
    V.build_html(out, g["players"], g["ball"], g["slot_team"],
                 [("model (frozen parameters)", pred), ("event-derived labels", letters[truth])],
                 C.STRICT_EXTERNAL_PITCH, title="Metrica sample game {}".format(a.external),
                 subtitle="public external data: model output against event-derived labels",
                 footer="Data: Metrica Sports public sample data "
                        "(github.com/metrica-sports/sample-data). In-sample: the shipped "
                        "parameters were trained on all three sample games.",
                 team_names=("home", "away"), frame_ids=g["frame"], step=a.step, fps=C.FPS)
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--team", default="YamineLamal", help="reads submission/<team>.csv")
    ap.add_argument("--external", type=int, choices=(1, 2, 3),
                    help="view a Metrica sample game instead of the competition match")
    ap.add_argument("--no-broadcast", action="store_true", help="do not show broadcast ball fill")
    ap.add_argument("--step", type=int, default=5, help="keep every Nth frame (default 5 = 5 fps)")
    ap.add_argument("--out", help="output HTML path")
    ap.add_argument("--no-open", action="store_true", help="do not open the browser")
    a = ap.parse_args()
    out = external(a) if a.external else competition(a)
    print("wrote {} ({:.1f} MB)".format(os.path.relpath(out, ROOT), os.path.getsize(out) / 1e6))
    if not a.no_open:
        webbrowser.open("file://" + os.path.abspath(out))
