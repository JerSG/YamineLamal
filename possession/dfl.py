"""DFL open data (IDSSE): seven Bundesliga / 2. Bundesliga matches, as extra external training data.

    Bassek, M., Rein, R., Weber, H., Memmert, D. (2025). An integrated dataset of spatiotemporal
    and event data in elite soccer. Scientific Data 12, 195. doi:10.1038/s41597-025-04505-y
    Data: Deutsche Fussball Liga (DFL), CC BY 4.0. https://doi.org/10.6084/m9.figshare.28196177

The positions files carry, on every ball frame, BallPossession (1 home, 2 guest) and BallStatus
(1 in play, 0 dead), recorded by the tracking provider. That is the same kind of label as the
competition's ground truth ("D where the corrected tracking marks the ball dead, otherwise the
team the tracking names"), so labels here are read directly, not derived from events.

The DFL ball is tracked almost always, unlike the competition's automatic tracking. To train
under realistic conditions, ball gaps are added by a two-state Markov chain whose probabilities
are measured on the Metrica external games (missing_mask_model), separately for live and dead
play, with a fixed seed per match. Nothing here reads competition data.
"""

import hashlib
import os
import xml.etree.ElementTree as ET

import numpy as np

H, A, D, X = 0, 1, 2, 3
SECTIONS = ("firstHalf", "secondHalf", "firstHalfExtra", "secondHalfExtra")


def _teams(info_path):
    root = ET.parse(info_path).getroot()
    home = guest = None
    for t in root.iter("Team"):
        if t.get("Role") == "home":
            home = t.get("TeamId")
        elif t.get("Role") == "guest":
            guest = t.get("TeamId")
    pitch = None
    env = root.find(".//Environment")
    if env is not None and env.get("PitchX"):
        pitch = (float(env.get("PitchX")), float(env.get("PitchY")))
    return home, guest, pitch or (105.0, 68.0)


def _read_positions(pos_path):
    """{(section, team, person): (N array, X array, Y array, possession, status)}, streamed."""
    out = {}
    key = None
    rows = None
    for event, el in ET.iterparse(pos_path, events=("start", "end")):
        if event == "start" and el.tag == "FrameSet":
            key = (el.get("GameSection"), el.get("TeamId"), el.get("PersonId"))
            rows = []
        elif event == "end" and el.tag == "Frame":
            rows.append((int(el.get("N")), float(el.get("X")), float(el.get("Y")),
                         int(el.get("BallPossession") or 0), int(el.get("BallStatus") or -1)))
            el.clear()
        elif event == "end" and el.tag == "FrameSet":
            arr = np.array(rows, dtype=np.float64)
            out[key] = arr
            el.clear()
            key, rows = None, None
    return out


def load_game(root, match_id):
    """One match in the common format of possession/external.py, plus labels (H/A/D/X)."""
    files = os.listdir(root)
    pos = [f for f in files if f.startswith("DFL_04_03_positions_raw_observed") and match_id in f]
    info = [f for f in files if f.startswith("DFL_02_01_matchinformation") and match_id in f]
    if not pos or not info:
        raise IOError("DFL match {} not found in {}".format(match_id, root))
    home, guest, pitch = _teams(os.path.join(root, info[0]))
    sets = _read_positions(os.path.join(root, pos[0]))
    length, width = pitch
    sections = [s for s in SECTIONS if any(k[0] == s for k in sets)]
    persons = sorted({(k[1], k[2]) for k in sets if k[1] in (home, guest)})
    slot = {p: i for i, p in enumerate(persons)}
    slot_team = np.array([H if t == home else A for t, _ in persons], dtype=np.int8)
    chunks_p, chunks_b, chunks_l, frame_ids, period = [], [], [], [], []
    for si, sec in enumerate(sections):
        ball_key = [k for k in sets if k[0] == sec and k[1] == "BALL"]
        if not ball_key:
            continue
        b = sets[ball_key[0]]
        n0, n1 = int(b[:, 0].min()), int(b[:, 0].max())
        T = n1 - n0 + 1
        players = np.full((T, len(persons), 3), np.nan, dtype=np.float32)
        for (s, team, person), arr in sets.items():
            if s != sec or (team, person) not in slot:
                continue
            i = (arr[:, 0] - n0).astype(int)
            ok = (i >= 0) & (i < T)
            players[i[ok], slot[(team, person)], 0] = (arr[ok, 1] + length / 2) / length
            players[i[ok], slot[(team, person)], 1] = (arr[ok, 2] + width / 2) / width
        ball = np.full((T, 3), np.nan, dtype=np.float32)
        lab = np.full(T, X, dtype=np.int8)
        i = (b[:, 0] - n0).astype(int)
        ball[i, 0] = (b[:, 1] + length / 2) / length
        ball[i, 1] = (b[:, 2] + width / 2) / width
        poss, status = b[:, 3].astype(int), b[:, 4].astype(int)
        lab[i] = np.where(status == 0, D, np.where(poss == 1, H, np.where(poss == 2, A, X)))
        # one ungraded separator frame between sections, so blocks never span a half-time
        if chunks_p:
            chunks_p.append(np.full((1, len(persons), 3), np.nan, dtype=np.float32))
            chunks_b.append(np.full((1, 3), np.nan, dtype=np.float32))
            chunks_l.append(np.array([X], dtype=np.int8))
            frame_ids.append(np.array([-1]))
            period.append(np.array([0]))
        chunks_p.append(players)
        chunks_b.append(ball)
        chunks_l.append(lab)
        frame_ids.append(np.arange(n0, n1 + 1))
        period.append(np.full(T, si + 1))
    return {"name": match_id, "frame": np.concatenate(frame_ids),
            "period": np.concatenate(period), "players": np.concatenate(chunks_p),
            "ball": np.concatenate(chunks_b), "slot_team": slot_team,
            "labels": np.concatenate(chunks_l), "pitch": pitch}


def load_all(root, cache_dir=None):
    """Every match in root, in id order; parsed XML cached as pickles in cache_dir if given."""
    import pickle
    out = []
    for mid in match_ids(root):
        path = os.path.join(cache_dir, mid + ".pkl") if cache_dir else None
        if path and os.path.exists(path):
            with open(path, "rb") as h:
                g = pickle.load(h)
        else:
            g = load_game(root, mid)
            if path:
                os.makedirs(cache_dir, exist_ok=True)
                with open(path, "wb") as h:
                    pickle.dump(g, h, protocol=4)
        g["name"] = mid
        out.append(g)
    return out


def match_ids(root):
    return sorted({f.split("_")[-1].replace(".xml", "") for f in os.listdir(root)
                   if f.startswith("DFL_04_03_positions_raw_observed")})


# ------------------------------------------------------------------------------------------------
# realistic ball gaps, learned from the Metrica external games
# ------------------------------------------------------------------------------------------------
def missing_mask_model(metrica_games, metrica_labels):
    """P(ball missing at t | missing or seen at t-1, live or dead at t), from the Metrica games."""
    counts = np.zeros((2, 2, 2))   # [prev_missing, live(0)/dead(1), now_missing]
    for g, lab in zip(metrica_games, metrica_labels):
        miss = ~np.isfinite(g["ball"][:, 0])
        graded = lab != X
        ok = graded[1:] & graded[:-1]
        prev, now = miss[:-1][ok], miss[1:][ok]
        dead = (lab[1:][ok] == D)
        np.add.at(counts, (prev.astype(int), dead.astype(int), now.astype(int)), 1)
    p_missing = counts[:, :, 1] / np.maximum(counts.sum(axis=2), 1)
    return {"p_missing_given_prev_seen": p_missing[0].tolist(),      # [live, dead]
            "p_missing_given_prev_missing": p_missing[1].tolist()}


def apply_missing(game, model, seed_text):
    """Copy of the game with ball gaps drawn from the model (fixed seed per match)."""
    seed = int(hashlib.sha256(seed_text.encode()).hexdigest()[:8], 16)
    rng = np.random.RandomState(seed)
    lab = game["labels"]
    p_seen = np.array(model["p_missing_given_prev_seen"])
    p_miss = np.array(model["p_missing_given_prev_missing"])
    u = rng.random_sample(len(lab))
    missing = np.zeros(len(lab), dtype=bool)
    prev = False
    for t in range(len(lab)):
        if lab[t] == X:
            prev = False
            continue
        dead = int(lab[t] == D)
        p = p_miss[dead] if prev else p_seen[dead]
        prev = bool(u[t] < p)
        missing[t] = prev
    g = dict(game)
    ball = game["ball"].copy()
    ball[missing] = np.nan
    g["ball"] = ball
    return g
