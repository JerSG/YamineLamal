"""Tests on synthetic data only -- no match data is needed or read.

    python -m pytest -q tests          # or:  python tests/test_possession.py
"""

import json
import os
import sys
import tempfile

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "evaluation"))

from possession import features as F  # noqa: E402
from possession import sources as S  # noqa: E402
from possession.pipeline import write_submission  # noqa: E402


def test_neutral_values_carry_no_evidence():
    """'Not measurable on this frame' adds nothing to any state's log-likelihood."""
    from possession import strict as ST
    f = F.pspeed_feature({"mean_speed": np.array([0.5, 1.0, np.nan, np.nan, np.nan])})
    assert list(f.codes) == [0, 1, 5, 5, 5]                     # 5 = "unknown"
    table = ST.PossessionHMM._tables_from(f.init_team, f.init_dead, f)
    trans = np.full((3, 3), 1.0 / 3)
    m = ST.FrozenHMM([f], [table], trans, 1.0, [(0, 5)])
    L = m.log_emissions()
    assert np.all(L[2:] == 0.0)
    assert not np.all(L[:2] == 0.0)


def test_run_lengths_and_gap_bins():
    missing = np.array([0, 1, 1, 0] + [1] * 27, dtype=bool)
    lengths = F.run_lengths(missing)
    assert lengths[1] == 2 and lengths[4] == 27 and lengths[0] == 1


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


def test_strict_external_model_trains_and_applies_on_synthetic_data():
    """Train on labelled synthetic 'external' games through the real training path (bins,
    team-shape model, tables, transitions, scale), then apply the result frozen to a new
    synthetic 'match' (no competition data anywhere)."""
    from possession import strict as ST
    n_side = 5
    slot_team = np.array([0] * n_side + [1] * n_side, dtype=np.int8)
    pitch = (105.0, 68.0)

    def game(seed, T=6000):
        r = np.random.RandomState(seed)
        lab = np.zeros(T, dtype=np.int8)
        t, team = 0, 0
        while t < T:
            n = r.randint(150, 450)
            lab[t:t + n] = team
            t += n
            if r.rand() < 0.3:
                m = r.randint(200, 500)
                lab[t:t + m] = 2
                t += m
            team = r.randint(0, 2) if lab[min(t, T) - 1] == 2 else 1 - team
        lab = lab[:T]
        P = 2 * n_side
        players = np.full((T, P, 3), np.nan, dtype=np.float32)
        base = r.uniform(0.3, 0.7, (P, 2))
        drift = np.cumsum(r.normal(0, 0.0008, (T, P, 2)) *
                          np.where(lab == 2, 0.2, 1.5)[:, None, None], 0)
        players[..., :2] = np.clip(base + drift, 0.02, 0.98)
        ball = np.full((T, 3), np.nan, dtype=np.float32)
        owner = np.where(lab == 1, n_side, 0)
        ball[:, :2] = players[np.arange(T), owner, :2] + 0.004
        ball[r.rand(T) < 0.35] = np.nan
        return {"name": "synthetic{}".format(seed), "players": players, "ball": ball,
                "slot_team": slot_team}, lab

    ext = [game(s) for s in (1, 2)]
    params = ST.train([g for g, _ in ext], [lab for _, lab in ext], "synthetic", verbose=False)
    assert set(params["features"]) == set(ST.MODEL_FEATURES)
    assert 5 < params["transition_stats"]["mean_alive_spell_s"] < 30
    params = json.loads(json.dumps(params))            # exactly what save/load round-trips
    g, lab = game(9)
    p, b = g["players"], g["ball"]
    blocks = [(0, len(lab))]
    events = [{"code": "SET PIECES", "type": "kick off (start)", "team": None, "start": 100,
               "end": 225}]
    path, model, forced = ST.apply(params, p, b, slot_team, events, len(lab), blocks, pitch)
    assert forced[:100].all() and not forced[100:].any() and (path[:100] == 2).all()
    assert np.mean(path[100:] == lab[100:]) > 0.6     # far above chance on an unseen 'match'
    # home/away symmetry: swapping the teams in the input mirrors every feature exactly, and
    # mirrors the output except where both teams are exactly equally likely -- there Viterbi's
    # fixed tie-break (first state wins) decides, deterministically
    swapped = (1 - slot_team).astype(np.int8)
    sm = params["shape_model"]
    state = {"w": np.array(sm["weights"]), "scale": np.array(sm["column_scale"])}
    with ST.use_bins(params["bins"]):
        f_orig = ST.model_features(ST.make_obs(p, b, slot_team, pitch, blocks), state)
        f_swap = ST.model_features(ST.make_obs(p, b, swapped, pitch, blocks), state)
    for fo, fs in zip(f_orig, f_swap):
        assert np.array_equal(fo.swap[fo.codes], fs.codes), fo.name
    path_sw, _, _ = ST.apply(params, p, b, swapped, events, len(lab), blocks, pitch)
    mirrored = np.where(path_sw == 0, 1, np.where(path_sw == 1, 0, path_sw))
    assert np.mean(path == mirrored) > 0.995
    # weights are a fixed local rule: one observation per run of identical codes
    w = ST.run_weights(np.array([0, 0, 0, 1, 2, 2]), [(0, 6)], min_look_frames=1)
    assert np.allclose(w, [1 / 3, 1 / 3, 1 / 3, 1, 0.5, 0.5])
    # ... capped at one observation per min_look_frames, however much the codes flicker
    flicker = np.tile([0, 1], 50)
    assert np.isclose(ST.run_weights(flicker, [(0, 100)], min_look_frames=25).sum(), 100 / 25.0)


def test_shape_model_is_antisymmetric_and_gaps_hit_their_share():
    """Swapping home and away negates the team-shape logit exactly, and every gap profile
    removes the documented share of each block."""
    from possession import degrade as DG
    from possession import shape
    from possession import strict as ST
    r = np.random.RandomState(4)
    T, P = 3000, 10
    st = np.array([0] * 5 + [1] * 5, dtype=np.int8)
    players = np.clip(np.cumsum(r.normal(0, 0.002, (T, P, 3)), 0) + 0.5, 0, 1).astype(np.float32)
    ball = players[:, 0].copy()
    pitch = (105.0, 68.0)
    o = ST.make_obs(players, ball, st, pitch, [(0, T)])
    o_sw = ST.make_obs(players, ball, (1 - st).astype(np.int8), pitch, [(0, T)])
    state = {"w": r.normal(0, 1, 25), "scale": np.ones(25)}
    z, z_sw = shape.logit(o, state), shape.logit(o_sw, state)
    ok = np.isfinite(z)
    assert ok.mean() > 0.9 and np.allclose(z[ok], -z_sw[ok])
    lab = np.zeros(T, dtype=np.int8)
    for prof in ("short", "long", "crowded"):
        g = DG.apply_profile({"name": "x", "players": players, "ball": ball, "labels": lab},
                             prof, 1)
        share = np.mean(~np.isfinite(g["ball"][:, 0]))
        assert abs(share - DG.MISSING_SHARE) < 0.002, (prof, share)


def test_restart_kick_is_dated_and_given_to_the_taker():
    """A ball still for 2 s next to a home player, then struck at 15 m/s: the second after the
    kick is 'kick_home'; no ball anywhere means no evidence."""
    from possession import deadball as DB
    from possession import strict as ST
    T = 200
    players = np.full((T, 2, 3), np.nan, dtype=np.float32)
    players[:, 0, :2] = (30.0 / 105, 10.0 / 68)          # home player at the ball
    players[:, 1, :2] = (60.0 / 105, 40.0 / 68)          # away player far away
    x = np.full(T, 30.5)
    x[50:] = 30.5 + 15.0 * (np.arange(T - 50) / 25.0)    # still until frame 50, then 15 m/s
    ball = np.stack([x / 105, np.full(T, 10.0 / 68), np.zeros(T)], 1).astype(np.float32)
    t = ST.summarise(players, ball, np.array([0, 1], dtype=np.int8), (105.0, 68.0))
    f = DB.restart_feature(t)
    marked = np.flatnonzero(f.codes == f.values.index("kick_home"))
    assert len(marked) == DB.KICK_MARK_FRAMES and 40 <= marked[0] <= 60
    assert not np.any(f.codes == f.values.index("kick_away"))
    ball[:] = np.nan
    f = DB.restart_feature(ST.summarise(players, ball, np.array([0, 1], dtype=np.int8),
                                        (105.0, 68.0)))
    assert np.all(f.neutral[f.codes])


def test_viewer_page_round_trips_tracking_and_labels():
    """The replay page embeds subsampled, quantised tracking and run-length labels, and nothing
    else is needed to open it."""
    import base64
    import json
    from possession import viewer as V
    T, slots = 101, 4
    players = np.random.RandomState(3).uniform(0, 1, (T, slots, 3)).astype(np.float32)
    players[7, 2] = np.nan
    ball = players[:, 0].copy()
    labels = np.array(["X"] * 10 + ["H"] * 40 + ["D"] * 20 + ["A"] * 31)
    with tempfile.TemporaryDirectory() as d:
        out = V.build_html(os.path.join(d, "v.html"), players, ball, [0, 0, 1, 1],
                           [("ours", labels)], (105.0, 68.0), "t", step=1)
        page = open(out, encoding="utf-8").read()
    assert "__DATA__" not in page and "http" not in page.split("<script>")[1]   # no network
    data = json.loads(page.split("const D = ", 1)[1].split(";\n", 1)[0])
    q = np.frombuffer(base64.b64decode(data["players"]), dtype="<u2").reshape(T, slots, 2)
    assert q[7, 2, 0] == V.NAN_CODE
    ok = np.isfinite(players[..., 0])
    assert np.allclose(q[ok] / 65534.0, players[..., :2][ok], atol=1e-4)
    rle = data["label_tracks"][0]["rle"]
    assert rle == [["X", 10], ["H", 40], ["D", 20], ["A", 31]]


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print("ok  " + name)
