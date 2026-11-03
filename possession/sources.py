"""Readers for the provided assets. Standard library + numpy only.

Three inputs, in the order the pipeline prefers them:

  read_atd()              the raw tactical tracking (players + ball), per video frame
  read_nearest_signal()   fallback when the raw tracking is absent: the organisers' published
                          `baseline_raw.csv`, which is *exactly* "team of the nearest tracked
                          player to the ball, D where the ball is missing", computed from the ATD
                          by starter/baseline.py. It contains no ground truth.
  read_smart_tagging()    the automatic tactical tagging, as video-frame intervals
"""

import os
import xml.etree.ElementTree as ET

import numpy as np

HOME, AWAY = 0, 1


# ------------------------------------------------------------------------------------------------
# ATD metadata
# ------------------------------------------------------------------------------------------------
def read_atd_meta(meta_path):
    """Team names, slot -> team, attack direction of each team in the first half."""
    root = ET.parse(meta_path).getroot()
    teams = list(root.iter("Team"))
    if len(teams) != 2:
        raise ValueError("{}: expected 2 teams, found {}".format(meta_path, len(teams)))
    # FIFA/EPTS convention (and kloppy's): the first team is home. Cross-checked against the
    # Score element's idLocalTeam when it is present.
    ids = [t.get("id") for t in teams]
    score = root.find(".//Score")
    if score is not None and score.get("idLocalTeam") in ids:
        if score.get("idLocalTeam") != ids[0]:
            teams = teams[::-1]
            ids = ids[::-1]
    names = [t.findtext("Name") for t in teams]
    direction = {}
    for side, team in enumerate(teams):
        for param in team.iter("ProviderParameter"):
            if param.findtext("Name") == "attack_direction_first_half":
                direction[side] = param.findtext("Value")
    slots = np.array([HOME if p.get("teamId") == ids[0] else AWAY for p in root.iter("Player")],
                     dtype=np.int8)
    if not len(slots):
        raise ValueError("{}: no Player entries".format(meta_path))
    return {"names": names, "slot_team": slots, "attack_direction_first_half": direction}


# ------------------------------------------------------------------------------------------------
# Raw ATD
# ------------------------------------------------------------------------------------------------
def read_atd(raw_path, n_slots, n_frames):
    """Arrays over video frames: players (T, n_slots, 3) as x, y, speed; ball (T, 3) as x, y, s.

    Line format, from the metadata's DataFormatSpecification:
        <frameCount>:<x,y,s>;<x,y,s>;...(n_slots)...:<ball x,y,s>
    Positions normalised 0-1, NaN when not observed. video_frame = frameCount - 1 (DATA.md).
    float32 keeps the full match at ~110 MB.
    """
    players = np.full((n_frames, n_slots, 3), np.nan, dtype=np.float32)
    ball = np.full((n_frames, 3), np.nan, dtype=np.float32)
    expected = 1 + 3 * n_slots + 3
    bad = 0
    with open(raw_path) as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            values = line.replace(":", ",").replace(";", ",").split(",")
            if len(values) != expected:
                bad += 1
                continue
            frame = int(values[0]) - 1
            if not 0 <= frame < n_frames:
                bad += 1
                continue
            row = np.array(values[1:], dtype=np.float32)
            players[frame] = row[:3 * n_slots].reshape(n_slots, 3)
            ball[frame] = row[3 * n_slots:]
    if bad:
        print("  warning: {} malformed ATD lines skipped".format(bad))
    return players, ball


# ------------------------------------------------------------------------------------------------
# Fallback: the published nearest-player signal
# ------------------------------------------------------------------------------------------------
def read_nearest_signal(path, n_frames):
    """Per-frame code: 0 home nearest, 1 away nearest, -1 ball not tracked.

    `baseline_raw.csv` is the unsmoothed output of the organisers' starter/baseline.py: for every
    frame with a tracked ball, the team of the nearest tracked player; `D` exactly where the ATD
    has no ball. It is computed from the ATD alone (see evaluation/results/README.md).
    """
    codes = np.full(n_frames, -1, dtype=np.int8)
    with open(path) as handle:
        header = handle.readline().strip().lower()
        if header != "frame,label":
            raise ValueError("{}: unexpected header {!r}".format(path, header))
        for i, line in enumerate(handle):
            frame, label = line.strip().split(",")
            if int(frame) != i:
                raise ValueError("{}: frame column out of order at row {}".format(path, i))
            codes[i] = {"H": HOME, "A": AWAY}.get(label.strip().upper(), -1)
    if i + 1 != n_frames:
        raise ValueError("{}: {} rows, expected {}".format(path, i + 1, n_frames))
    return codes


# ------------------------------------------------------------------------------------------------
# Smart Tagging
# ------------------------------------------------------------------------------------------------
def team_side(name, home_name="Barcelona", away_name="Atletic Club"):
    """Map a tag's Team value to HOME / AWAY / None. Case- and spelling-tolerant."""
    if not name:
        return None
    n = name.strip().lower()
    if n in ("n/a", "na", "none", ""):
        return None
    if n == home_name.strip().lower() or "barcelona" in n:
        return HOME
    if n == away_name.strip().lower() or "athletic" in n or "atletic" in n:
        return AWAY
    return None


def read_smart_tagging(path, fps=25, home_name="Barcelona", away_name="Atletic Club"):
    """[{code, type, team, start, end}] with frames half-open [start, end), sorted by start."""
    out = []
    for inst in ET.parse(path).iter("instance"):
        tags = {lab.findtext("group"): lab.findtext("text") for lab in inst.findall("label")}
        out.append({
            "code": (inst.findtext("code") or "").strip().upper(),
            "type": (tags.get("Type") or "").strip().lower(),
            "team": team_side(tags.get("Team"), home_name, away_name),
            "start": int(round(float(inst.findtext("start")) * fps)),
            "end": int(round(float(inst.findtext("end")) * fps)),
        })
    out.sort(key=lambda e: (e["start"], e["end"]))
    return out


def file_exists(path):
    return bool(path) and os.path.isfile(path)
