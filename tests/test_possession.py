"""Tests on synthetic data only -- no match data is needed or read.

    python -m pytest -q tests          # or:  python tests/test_possession.py
"""

import os
import sys
import tempfile

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "evaluation"))

from possession import features as F  # noqa: E402
from possession import sources as S  # noqa: E402
from possession.hmm import PossessionHMM, H, A, D, mean_run_length  # noqa: E402
from possession.pipeline import write_submission  # noqa: E402


def planted_match(seed=0, n_spells=60):
    """A synthetic half: alternating H/A spells with stoppages, and noisy observations of it."""
    rng = np.random.RandomState(seed)
    truth = []
    team = H
    for _ in range(n_spells):
        truth += [team] * rng.randint(150, 600)
        if rng.rand() < 0.3:
            truth += [D] * rng.randint(200, 500)
            team = rng.choice([H, A])
        else:
            team = A if team == H else H
    truth = np.array(truth)
    T = len(truth)
    # nearest team: right 70% of the time while alive, noise when dead; in runs of 10 frames
    nearest = np.empty(T, dtype=np.int8)
    for s in range(0, T, 10):
        st = truth[s]
        if st == D:
            v = rng.choice([-1, 0, 1], p=[0.6, 0.2, 0.2])
        else:
            right = 0 if st == H else 1
            v = rng.choice([right, 1 - right, -1], p=[0.6, 0.25, 0.15])
        nearest[s:s + 10] = v
    # player speed: slow when dead
    speed = np.where(truth == D, rng.gamma(2.0, 0.35, T), rng.gamma(4.0, 0.5, T))
    speed = F.centred_mean(speed, 12)
    return truth, nearest, speed


def build(nearest, speed):
    ball = F.ball_feature_basic(nearest)
    ps = F.pspeed_feature({"mean_speed": speed})
    return [ball, ps]


def test_hmm_recovers_planted_possessions():
    truth, nearest, speed = planted_match()
    blocks = [(0, len(truth))]
    model = PossessionHMM(build(nearest, speed), blocks=blocks)
    model.calibrate_scale(blocks, verbose=False)
    model.fit(blocks, verbose=False)
    model.calibrate_scale(blocks, verbose=False)
    path = model.decode(blocks)
    acc = np.mean(path == truth)
    assert acc > 0.85, acc
    # dead balls found from player speed alone
    dead_recall = np.mean(path[truth == D] == D)
    assert dead_recall > 0.8, dead_recall


def test_home_away_symmetry():
    """Swapping the teams in the input must swap H and A in the output, frame for frame."""
    truth, nearest, speed = planted_match(seed=3, n_spells=30)
    swapped = np.where(nearest >= 0, 1 - nearest, nearest).astype(np.int8)
    blocks = [(0, len(truth))]
    paths = []
    for obs in (nearest, swapped):
        m = PossessionHMM(build(obs, speed), blocks=blocks)
        m.fit(blocks, verbose=False)
        paths.append(m.decode(blocks))
    mirrored = np.where(paths[1] == H, A, np.where(paths[1] == A, H, D))
    assert np.mean(paths[0] == mirrored) > 0.999


def test_neutral_values_carry_no_evidence():
    codes = np.array([0, 1, 5, 5, 5])  # 5 = "unknown"
    f = F.pspeed_feature({"mean_speed": np.array([0.5, 1.0, np.nan, np.nan, np.nan])})
    assert list(f.codes) == list(codes)
    m = PossessionHMM([f], blocks=[(0, 5)])
    L = m.log_emissions()
    assert np.all(L[2:] == 0.0)
    assert not np.all(L[:2] == 0.0)


def test_run_lengths_and_gap_bins():
    missing = np.array([0, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                        1, 1, 1, 1, 1, 1, 1], dtype=bool)
    lengths = F.run_lengths(missing)
    assert lengths[1] == 2 and lengths[4] == 27 and lengths[0] == 1
    assert mean_run_length(np.array([0, 0, 1, 1, 1, 2])) == 2.0


def write_synthetic_atd(folder, n_frames=200, n_slots=4):
    """A metadata XML + raw file in the exact layout of the provided ATD."""
    meta = os.path.join(folder, "m.xml")
    with open(meta, "w") as h:
        h.write('<main><Metadata><Sessions><Session><MatchParameters>'
                '<Score idLocalTeam="T1" idVisitingTeam="T2"/></MatchParameters></Session>'
                '</Sessions><Teams>'
                '<Team id="T1"><Name>Barcelona</Name><ProviderTeamsParameters><ProviderParameter>'
                '<Name>attack_direction_first_half</Name><Value>left_to_right</Value>'
                '</ProviderParameter></ProviderTeamsParameters></Team>'
                '<Team id="T2"><Name>Atletic Club</Name></Team></Teams><Players>')
        for i in range(n_slots):
            h.write('<Player id="p{0}" teamId="{1}"><Name>Track {0}</Name></Player>'.format(
                i + 1, "T1" if i < n_slots // 2 else "T2"))
        h.write('</Players></Metadata></main>')
    raw = os.path.join(folder, "r.txt")
    with open(raw, "w") as h:
        for f in range(1, n_frames + 1):
            groups = []
            for i in range(n_slots):
                if i == 3 and f % 2:
                    groups.append("NaN,NaN,NaN")
                else:
                    groups.append("{:.5f},{:.5f},{:.3f}".format(0.1 + 0.2 * i, 0.5, 1.5))
            ball = "NaN,NaN,NaN" if f % 5 == 0 else "0.10500,0.50000,NaN"
            h.write("{}:{}:{}\n".format(f, ";".join(groups), ball))
    return meta, raw


def test_atd_reader_and_tracking_summary():
    with tempfile.TemporaryDirectory() as d:
        meta_path, raw_path = write_synthetic_atd(d)
        meta = S.read_atd_meta(meta_path)
        assert meta["names"] == ["Barcelona", "Atletic Club"]
        assert list(meta["slot_team"]) == [0, 0, 1, 1]
        assert meta["attack_direction_first_half"][S.HOME] == "left_to_right"
        players, ball = S.read_atd(raw_path, 4, 200)
        assert players.shape == (200, 4, 3) and ball.shape == (200, 3)
        assert np.isnan(players[0, 3, 0]) and not np.isnan(players[1, 3, 0])  # frame_id 1 -> row 0
        t = F.tracking_summary(players, ball, meta["slot_team"], (105.3, 68.0))
        seen = ~np.isnan(ball[:, 0])
        assert np.all(t["nearest"][seen] == S.HOME)  # ball sits on slot 0 (home)
        assert np.all(t["nearest"][~seen] == -1)
        assert np.allclose(t["dh"][seen], 0.005 * 105.3, atol=1e-3)  # 0.105 vs 0.100 in x
        fb = F.ball_feature_full(t)
        assert set(np.array(fb.values)[fb.codes[seen]]) == {"home_on"}


def test_submission_format_passes_official_validator():
    import pandas as pd
    import possession_metrics as pm
    labels = np.array(["X"] * 199319)
    labels[11185:90652] = "H"
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "t.csv")
        write_submission(path, labels)
        frame = pm.read_label_csv(path)
        assert pm.validate_submission(frame, 199319) is None
        assert len(pd.read_csv(path)) == 199319


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print("ok  " + name)
