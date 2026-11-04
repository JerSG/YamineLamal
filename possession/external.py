"""External training data: Metrica Sports' public sample games (other matches).

    git clone https://github.com/metrica-sports/sample-data external/metrica-sample-data
    python scripts/train_external.py          # -> models/external_params.json

Used only by the strict-external model (pipeline --strict-external): every parameter that is
learned or tuned comes from these games, never from the competition match, its ground truth or
its video (organisers' clarification, "Training and tuning").

Source: https://github.com/metrica-sports/sample-data -- three anonymised professional matches with
tracking (players + ball, 25 fps, coordinates 0-1) and manually collected event data. No formal
licence file; the README provides the data for the analytics community and asks that public use
acknowledge the source, which the technical report does.

Labels are derived from the events, mechanically:

  * alive, team T   from a possession event of T (PASS, RECOVERY, SET PIECE, SHOT, CARRY) until
                    the next possession event of the other team (contested events -- CHALLENGE,
                    BALL LOST -- do not change possession by themselves, matching the challenge's
                    "controlled possession" rule)
  * dead            from BALL OUT, a foul (FAULT RECEIVED) or a goal until the next SET PIECE,
                    and before each period's kickoff
  * X (not graded)  outside the periods
"""

import json
import os
import xml.etree.ElementTree as ET

import numpy as np

H, A, D, X = 0, 1, 2, 3
POSSESSION = ("PASS", "RECOVERY", "SET PIECE", "SHOT", "CARRY")
STOPPAGE = ("BALL OUT", "FAULT RECEIVED")


# ------------------------------------------------------------------------------------------------
# readers
# ------------------------------------------------------------------------------------------------
def _read_metrica_csv(path):
    """(period, frame, players (T, P, 2), ball (T, 2)) from one team's RawTrackingData CSV."""
    with open(path) as h:
        rows = h.read().splitlines()
    header = rows[2].split(",")
    body = [r.split(",") for r in rows[3:] if r]
    period = np.array([int(r[0]) for r in body])
    frame = np.array([int(r[1]) for r in body])
    values = np.array([[float(v) if v not in ("", "NaN") else np.nan for v in r[3:]]
                       for r in body], dtype=np.float32)
    n_cols = len(header) - 3
    values = values[:, :n_cols]
    players = values[:, :-2].reshape(len(body), -1, 2)
    ball = values[:, -2:]
    return period, frame, players, ball


def load_csv_game(root, game):
    """Sample games 1 and 2 (Metrica CSV): one dict in the common format below."""
    d = os.path.join(root, "Sample_Game_{}".format(game))
    stem = "Sample_Game_{}".format(game)
    period, frame, home, ball = _read_metrica_csv(
        os.path.join(d, stem + "_RawTrackingData_Home_Team.csv"))
    _, frame_a, away, _ = _read_metrica_csv(
        os.path.join(d, stem + "_RawTrackingData_Away_Team.csv"))
    if not np.array_equal(frame, frame_a):
        raise ValueError("game {}: home and away tracking frames differ".format(game))
    players = np.concatenate([home, away], axis=1)
    slot_team = np.array([H] * home.shape[1] + [A] * away.shape[1], dtype=np.int8)
    events = []
    with open(os.path.join(d, stem + "_RawEventsData.csv")) as h:
        cols = h.readline().strip().split(",")
        for line in h:
            r = dict(zip(cols, line.strip().split(",")))
            events.append({"team": H if r["Team"] == "Home" else A,
                           "type": r["Type"].strip().upper(),
                           "subtype": (r.get("Subtype") or "").strip().upper(),
                           "start": int(r["Start Frame"]), "end": int(r["End Frame"]),
                           "period": int(r["Period"])})
    return _common("Sample_Game_{}".format(game), frame, period, players, ball, slot_team,
                   events)


def load_epts_game(root, game=3):
    """Sample game 3 (FIFA EPTS + JSON events), the same format as the competition data."""
    d = os.path.join(root, "Sample_Game_{}".format(game))
    stem = "Sample_Game_{}".format(game)
    meta = ET.parse(os.path.join(d, stem + "_metadata.xml")).getroot()
    teams = [t.get("id") for t in meta.iter("Team")]
    player_ids = [p.get("id") for p in meta.iter("Player")]
    column = {pid: i for i, pid in enumerate(player_ids)}
    slot_team = np.array([H if p.get("teamId") == teams[0] else A for p in meta.iter("Player")],
                         dtype=np.int8)
    channel_player = {c.get("id"): c.get("playerId") for c in meta.iter("PlayerChannel")}
    # One DataFormatSpecification per stretch between substitutions: which players, in order.
    specs = []
    for spec in meta.iter("DataFormatSpecification"):
        order = []
        for ref in spec.iter("PlayerChannelRef"):
            pid = channel_player[ref.get("playerChannelId")]
            if not order or order[-1] != pid:
                order.append(pid)
        specs.append((int(spec.get("startFrame")), int(spec.get("endFrame")),
                      np.array([column[p] for p in order])))
    specs.sort()
    frames, ball, rows = [], [], []
    with open(os.path.join(d, stem + "_tracking.txt")) as h:
        for line in h:
            head, ppart, bpart = line.strip().split(":")
            frames.append(int(head))
            rows.append([[float(v) for v in g.split(",")] for g in ppart.split(";")])
            b = bpart.split(",")
            ball.append([float(b[0]), float(b[1])])
    frames = np.array(frames)
    players = np.full((len(frames), len(player_ids), 2), np.nan, dtype=np.float32)
    si = 0
    for i, f in enumerate(frames):
        while si < len(specs) - 1 and f > specs[si][1]:
            si += 1
        cols = specs[si][2]
        if len(rows[i]) != len(cols):
            raise ValueError("game {} frame {}: {} players on the line, {} in its format "
                             "specification".format(game, f, len(rows[i]), len(cols)))
        players[i, cols] = rows[i]
    ball = np.array(ball, dtype=np.float32)
    with open(os.path.join(d, stem + "_events.json")) as h:
        raw = json.load(h)["data"]
    team_ids = {}
    events = []
    for e in raw:
        tid = (e.get("team") or {}).get("id")
        if tid not in team_ids:
            team_ids[tid] = H if tid == teams[0] else A
        sub = e.get("subtypes")
        if isinstance(sub, list):
            sub = " ".join(s.get("name", "") for s in sub)
        elif isinstance(sub, dict):
            sub = sub.get("name", "")
        events.append({"team": team_ids[tid], "type": e["type"]["name"].strip().upper(),
                       "subtype": (sub or "").strip().upper(),
                       "start": int(e["start"]["frame"]), "end": int(e["end"]["frame"]),
                       "period": int(e["period"])})
    # periods from the events (EPTS metadata carries them too, but events are authoritative here)
    period = np.zeros(len(frames), dtype=int)
    for p in sorted({e["period"] for e in events}):
        fs = [e["start"] for e in events if e["period"] == p] + \
             [e["end"] for e in events if e["period"] == p]
        period[(frames >= min(fs)) & (frames <= max(fs))] = p
    return _common(stem, frames, period, players, ball, slot_team, events)


def _common(name, frame, period, players_xy, ball_xy, slot_team, events):
    """Common format: players (T, P, 3) as x, y, speed (NaN: computed later from positions),
    ball (T, 3), period per row, frame id per row, events sorted by start."""
    T = len(frame)
    players = np.full((T, players_xy.shape[1], 3), np.nan, dtype=np.float32)
    players[..., :2] = players_xy
    ball = np.full((T, 3), np.nan, dtype=np.float32)
    ball[:, :2] = ball_xy
    events.sort(key=lambda e: (e["start"], e["end"]))
    return {"name": name, "frame": np.asarray(frame), "period": np.asarray(period),
            "players": players, "ball": ball, "slot_team": slot_team, "events": events}


# ------------------------------------------------------------------------------------------------
# labels from events
# ------------------------------------------------------------------------------------------------
def labels(game):
    """Per-row label H/A/D/X from the events (see module docstring)."""
    frame, period = game["frame"], game["period"]
    row_of = {int(f): i for i, f in enumerate(frame)}
    T = len(frame)
    out = np.full(T, X, dtype=np.int8)
    for p in sorted(set(period.tolist()) - {0}):
        rows = np.flatnonzero(period == p)
        evs = [e for e in game["events"] if e["period"] == p]
        kick = [e["start"] for e in evs if e["type"] == "SET PIECE" and "KICK OFF" in e["subtype"]]
        state = D
        team = None
        start_f = int(min(kick)) if kick else int(frame[rows[0]])
        # change points: (frame, new_state, team)
        changes = [(int(frame[rows[0]]), D, None)]
        for e in evs:
            if e["type"] in POSSESSION:
                changes.append((e["start"], "POSS", e["team"]))
            elif e["type"] in STOPPAGE:
                changes.append((e["start"], D, None))
            if e["type"] == "SHOT" and "GOAL" in e["subtype"] and "OFF" not in e["subtype"]:
                changes.append((e["end"], D, None))
        changes.sort(key=lambda c: c[0])
        cur_state, cur_team = D, None
        ci = 0
        for i in rows:
            f = int(frame[i])
            while ci < len(changes) and changes[ci][0] <= f:
                fs, st, tm = changes[ci]
                if fs >= start_f:
                    if st == "POSS":
                        cur_state, cur_team = "ALIVE", tm
                    else:
                        cur_state = D
                ci += 1
            if f < start_f or cur_state == D:
                out[i] = D
            else:
                out[i] = cur_team
    return out


def load_all(root):
    return [load_csv_game(root, 1), load_csv_game(root, 2), load_epts_game(root, 3)]
