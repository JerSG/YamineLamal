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


def test_broadcast_alignment_recovers_offset_and_rejects_garbage():
    """A synthetic 'broadcast' = a partial, noisy view of the tactical players at a known offset,
    plus one shot of unrelated positions. The aligner must find the offset, fill only missing
    tactical balls, and reject the garbage shot."""
    from possession import broadcast as B
    rng = np.random.RandomState(1)
    T, n_slots, pitch = 3000, 8, (105.3, 68.0)
    slot_team = np.array([0, 0, 0, 0, 1, 1, 1, 1], dtype=np.int8)
    # tactical: players on slow random walks, ball near player 0, missing every 3rd frame
    walk = np.cumsum(rng.normal(0, 0.002, (T, n_slots, 2)), axis=0) + rng.uniform(0.2, 0.8,
                                                                               (1, n_slots, 2))
    tac_players = np.concatenate([walk, np.ones((T, n_slots, 1))], axis=2).astype(np.float32)
    tac_ball = np.full((T, 3), np.nan, dtype=np.float32)
    tac_ball[:, :2] = walk[:, 0] + 0.003
    tac_ball[::3] = np.nan
    # broadcast: frames 0..1199 show tactical frames 700..1899 (offset +700), 6 of 8 players
    offset, n_bc = 700, 1500
    bc_players = np.full((n_bc, n_slots, 3), np.nan, dtype=np.float32)
    bc_players[:1200, :6, :2] = walk[offset:offset + 1200, :6] + rng.normal(0, 0.002, (1200, 6, 2))
    bc_players[1200:, :6, :2] = rng.uniform(0, 1, (300, 6, 2))   # garbage shot
    bc_ball = np.full((n_bc, 3), np.nan, dtype=np.float32)
    bc_ball[:1200, :2] = walk[offset:offset + 1200, 0] + 0.003
    aim = np.zeros((n_bc, 2))
    aim[1200:] = 0.9          # a cut between the two shots
    ok = np.ones(n_bc, dtype=bool)
    tac_of_bc, table = B.align(bc_players, bc_ball, slot_team, tac_players, tac_ball, slot_team,
                               aim, ok, pitch, verbose=False)
    good = [r for r in table if r["start"] == 0][0]
    assert good["offset"] == offset and good["accepted"], good
    assert good["ball_median_err_m"] < 0.5
    bad = [r for r in table if r["start"] == 1200][0]
    assert not bad["accepted"], bad
    assert np.all(tac_of_bc[:1200] == np.arange(1200) + offset) and np.all(tac_of_bc[1200:] == -1)
    filled, mask = B.fill_ball(tac_ball, bc_ball, tac_of_bc)
    seen = np.isfinite(tac_ball[:, 0])
    assert not mask[seen].any()                                   # never overrides a real ball
    assert np.array_equal(filled[seen], tac_ball[seen], equal_nan=True)
    assert mask[offset:offset + 1200][~seen[offset:offset + 1200]].all()   # fills every gap


def test_video_addon_geometry_filters_and_safety():
    """Pure-numpy parts of the optional video step (no torch/OpenCV needed)."""
    from possession import video as V
    pitch = (105.3, 68.0)
    # tiles cover a 1920x1080 frame completely, with overlap
    cover = np.zeros((1080, 1920), dtype=bool)
    for x, y in V.tiles(1920, 1080):
        cover[y:y + V.TILE, x:x + V.TILE] = True
    assert cover.all()
    # pixels -> pitch inverts a pitch -> pixel homography
    H = np.array([[1500.0, -300.0, 200.0], [10.0, 600.0, 150.0], [0.0, -0.3, 1.0]])
    px = H @ np.array([0.3, 0.6, 1.0])
    x, y = V.pixels_to_pitch(H, px[0] / px[2], px[1] / px[2])
    assert abs(x - 0.3) < 1e-9 and abs(y - 0.6) < 1e-9
    # an isolated detection 50 m away from both neighbours is dropped, plausible ones kept
    pts = {100: (0.50, 0.50, 0.9), 105: (0.98, 0.50, 0.9), 110: (0.505, 0.50, 0.9)}
    kept = V.drop_isolated_jumps(pts, pitch)
    assert 105 not in kept and 100 in kept and 110 in kept
    # interpolation fills short, close gaps only
    xy = V.interpolate({100: (0.5, 0.5, 0.9), 105: (0.52, 0.5, 0.9), 200: (0.9, 0.5, 0.9)},
                       300, pitch)
    assert np.isfinite(xy[101:105, 0]).all() and np.isnan(xy[106:200, 0]).all()
    # the video never overrides a tracked ball
    ball = np.full((300, 3), np.nan)
    ball[102] = [0.1, 0.1, np.nan]
    filled, gap = V.fill_from_video(ball, {100: (0.5, 0.5, 0.9), 105: (0.52, 0.5, 0.9)},
                                    300, pitch)
    assert filled[102, 0] == 0.1 and not gap[102] and gap[101] and gap[103]
    # the acceptance check fails a detector that is far from the tracker, passes a close one
    trk = np.full((5000, 3), np.nan)
    trk[:, 0], trk[:, 1] = 0.5, 0.5
    checks = np.arange(0, 5000, 10)
    near = {int(f): (0.5 + 0.005, 0.5, 0.8) for f in checks}       # 0.5 m off
    far = {int(f): (0.5 + 0.1, 0.5, 0.8) for f in checks}          # 10.5 m off
    assert V.check_against_tracker(near, trk, checks, pitch)["passed"]
    assert not V.check_against_tracker(far, trk, checks, pitch)["passed"]
    # files round-trip, and the pipeline ignores a failed or trial check
    import tempfile
    from possession import pipeline as P
    with tempfile.TemporaryDirectory() as d:
        V.write_outputs(d, near, {"passed": True, "median_error_m": 0.5}, {"weights": "x"})
        points, stats = V.read_outputs(d)
        assert points == near and stats["passed"]
    assert V.read_outputs(os.path.join(ROOT, "no_such_folder")) == (None, None)
    assert P.video_hook(None, verbose=False) is None or os.path.isdir(
        os.path.join(ROOT, "data", "derived"))


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print("ok  " + name)
