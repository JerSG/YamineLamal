---
title: "Ball possession from automatic tracking, trained on other matches"
subtitle: "More than a Hack 2027, Phase 1: technical document"
author: "YamineLamal"
date: "November 2026"
---

# 1. Summary

We label every frame of FC Barcelona vs Athletic Club `H`, `A` or `D` with a **three-state hidden
Markov model** (home in possession, away in possession, dead ball) decoded by Viterbi.

**Every learned or tuned number comes from other matches.** The model is trained on Metrica Sports'
three public sample games, with labels derived mechanically from their event data. It is then
frozen and applied unchanged to the competition match. That covers the feature bin edges, the
emission probabilities, the transition probabilities and the one evidence scale. The competition
match's automatic tracking, Smart Tagging and broadcast recording are used **as input only**.
No ground truth, no video and no human-produced data about the match are used to fit, tune or
check anything. This follows the organisers' rules clarification: *"If you need to train (or
retrain), you MUST use only external data."*

Main components:

* **Two-camera fusion.** The broadcast recording is aligned to the tactical timeline
  automatically, shot by shot, by matching player positions (no mapping is provided). Its ball
  then fills tactical frames where the tracker lost the ball.
* **Eight per-frame categorical features**: five physical ones from the tracking and three from
  the Smart Tagging.
* **A cap on evidence:** each feature counts as one observation per run of identical values, and
  never more than one per second. This keeps the frozen model stable on tracking noisier than
  the data it was trained on.
* **Generalisation between matches:** trained on two external games and tested on the unseen
  third, spellF1 is **0.572 / 0.576 / 0.567** (macroF1 0.789 / 0.778 / 0.816), against that
  game's event-derived labels.

There is no competition score to report: the ground truth is private, and we did not evaluate the
submitted model against the public 5-minute window (Section 7).

# 2. Why this approach

The ranking metric, spellF1, matches *possessions* one-to-one by temporal overlap. The expensive
mistakes are therefore segmentation errors: a possession split in two by a flicker, two merged
across a turnover, a stoppage missed. A per-frame classifier followed by smoothing has no notion
of a segment. An HMM decoded with Viterbi does: it weighs each frame's evidence against the cost
of a transition, so a short burst of contrary evidence cannot split a possession unless it is
strong enough to pay for two transitions.

The rules require training and tuning on external data only. A small, interpretable model with
few parameters transfers between matches far more safely than a large one: eight categorical
features, a 3×3 transition matrix and one scale, all estimated on three external games. The
leave-one-game-out results in Section 6 show the transfer.

# 3. Assets: used and dropped

| asset | used for | notes |
|---|---|---|
| tactical ATD `_atd.txt` + `.xml` (competition) | input: player and ball positions, which slot belongs to which team | player speeds are recomputed from positions, as the external games have no speed channel |
| Smart Tagging (competition) | input: open-play phases, set pieces, goals, kickoffs | fixed prior tables, nothing learned |
| broadcast ATD + homography (competition) | input: ball positions for tactical frames without a ball (Section 4.6) | the homography is used only to find camera cuts |
| Metrica sample games 1-3 (external) | **all training and tuning** | https://github.com/metrica-sports/sample-data |
| tactical homography (competition) | **dropped** | a camera-zone feature was considered and dropped: no external equivalent to learn it from |
| tactical and broadcast videos | **dropped** | see 4.7 |
| public ground truth | **not used** for the submitted model | Section 7 |

**Data checks.** The tactical ATD has 199,319 contiguous records. The organisers' own
`starter/baseline.py`, run on our copy, reproduces their four published baseline files byte for
byte, which confirms the file and the `video_frame = frame_id - 1` alignment. The broadcast ATD
has 164,488 contiguous records, matching DATA.md.

# 4. Method

## 4.1 External training data and labels

Metrica Sports' public sample games are three anonymised professional matches, about 4.8 hours
(429,923 frames) at 25 fps, with tracking of players and ball and with event data. Games 1 and 2 come in
Metrica's CSV format; game 3 is in FIFA EPTS, the competition's format, which we read with its
per-substitution data-format sections. The ball is tracked on 61%, 59% and 68% of their graded
frames, similar to the competition match.

Labels are derived from the events mechanically (`possession/external.py`):

* **alive, team T:** from a possession event of T (pass, recovery, set piece, shot, carry) until
  the next possession event of the other team. Contested events (challenge, ball lost) do not
  change possession by themselves, matching the challenge's "controlled possession" rule;
* **dead:** from a ball-out, a foul or a goal until the next set piece, and before each period's
  kickoff.

The result looks like football: 294-343 possessions per game with a mean of 11.7-13.7 s, and
75-78 stoppages.

## 4.2 Features (one categorical value per frame)

| feature | values | what it measures |
|---|---|---|
| `ball` | home on ball (< 2 m), home nearest, away on, away nearest, contested, or 4 bins of ball-missing stretch length | who controls the ball, and how long it has been lost |
| `pspeed` | 5 bins of mean player speed | stoppages are walked |
| `bspeed` | 4 bins of ball speed | a placed ball is slow |
| `bpos` | in, edge, out, lost right after going out | ball against the touch and goal lines |
| `spread` | home wider, even, away wider | the team in possession opens the pitch |
| `phase` | none, home, away, both | Smart Tagging phases mapped to "team attacking" (Build Up, Progression, Creating Chances, … name the attacker; Defending tags name the defender) |
| `setpiece` | taker (home / away / not tagged) × quarter of the tag | set-piece windows |
| `goal` | time left until the kickoff after a goal | celebration and restart |

**Bin edges are learned on the external games** as equal-share quantiles (a fixed rule set before
seeing any data). Ball-missing stretch length uses the quartiles 19 / 33 / 53 s. Mean player
speed uses the fifths 1.0 / 1.4 / 1.9 / 2.5 m/s. Ball speed uses the quartiles 2.7 / 6.2 /
11.1 m/s. The spread threshold is the third of |log ratio| (0.35). The only physical constants are
general football facts, not tuned on any data: a player is "on the ball" within 2 m, and the ball
is out once 0.5 m over a line.

## 4.3 Model

States `H`, `A` and `D`. Emissions are a product over features:

    log p(o_t | s) = scale · Σ_f  w_{f,t} · log B_f[s, o_{f,t}]

* **Home/away mirror tie.** `H` and `A` share one table with values swapped home↔away, so a
  feature means "the team in possession" rather than "Barcelona". `D` is mirror-symmetric.
  Tables learned on anonymised external teams therefore apply to any two teams.
* **Neutral values.** "Not measurable on this frame" carries a likelihood ratio of 1.
* **Weights:** `w_{f,t} = 1 / max(run length, 25 frames)`. Each run of identical values of a
  feature counts as one observation, and no feature can contribute more than one observation per
  second. Without the cap, noisier tracking (many short runs) means more evidence, and our
  external experiments showed a sharp drop into fragmentation just above the best scale (spellF1
  from 0.50 to 0.41 between scales 3.4 and 3.6). With the cap, the curve is smooth (Section 6).

## 4.4 Training on the external games

All estimated on games 1-3, then frozen in `models/external_params.json`:

1. **Emission tables:** supervised counts per state from the event-derived labels, tied as above,
   with a mild Dirichlet prior. They are physically sensible. Players are under 1 m/s on 46.8% of
   dead frames against 8.4% of live ones. Among frames where the ball's position is known or it was
   last seen over a line, "lost right after going out" is 84.4% of dead frames against 5.2% of
   live ones. A ball-missing stretch covers 90%
   of dead frames against 15% of live ones.
2. **Transitions**, measured from the labels' own spells: mean possession 12.5 s, 24% of
   possessions ending in a stoppage, mean stoppage 22.7 s.
3. **Evidence scale:** the value on a fixed grid with the best mean spellF1 over the three games,
   computed with the organisers' own `possession_metrics.py`: **6.0**.

## 4.5 Applying the frozen model to the competition match

The same features are computed on the match with the frozen bin edges and decoded with the frozen
tables, transitions and scale. Nothing is re-estimated. Smart Tagging has no external counterpart,
so its three features use **fixed prior tables**, which are not learned. Set-piece tags are split
into quarters of their own length, a fixed rule. **Kickoffs:** frames of a graded half before its
kickoff tag opens can only be `D`, which covers the graded dead frames before the second-half
kickoff that FAQ.md mentions. The tag's own start is the boundary, and no offset is measured on
the match. Viterbi then runs per graded half; ungraded frames are written as `X`, and graded
frames are never `X`.

## 4.6 Two-camera fusion (input processing; nothing is learned)

The broadcast recording has its own timeline with cuts and replays, and no mapping is provided.
Within one continuous camera shot both recordings run at 25 fps, so a shot maps onto the tactical
video by one frame offset (`possession/broadcast.py`):

1. **Shots:** runs of frames with a valid broadcast homography and at least 4 tracked players,
   split where the camera's aim point jumps more than 4 m of pitch between two frames. There are
   160 shots of at least 2 s.
2. **Offset:** for 6 sample frames, the mean distance from each broadcast player to the nearest
   same-team tactical player (capped at 5 m), minimised over every offset of the match.
3. **Acceptance:** a mean fit under 1.5 m, at least 1 m better than any offset more than 2 s away,
   and an **independent check on the ball**, which the offset search never sees. Where both
   trackers have the ball, the positions must agree to a median of 2 m.

110 of 160 shots are accepted. Their ball positions agree to a median of **0.76 m**. Live play
aligns at a near-constant offset (+4011…+4017 frames in the first half, +26916…+26922 in the
second), and the exceptions are replays, each matched to the moment it shows. Where the tactical
ball is missing, an aligned broadcast ball fills it (**4,763** frames). An observed tactical ball
is never overridden. Every threshold was fixed before the first alignment ran.

## 4.7 Considered and not used: ball detection in the video

We also built a ball detector for the video, using a public model pre-trained on external images.
It could not be validated on the external games, which have no video. We kept only components
whose benefit we could measure without the competition ground truth, so it is not part of the
submission or the repository.

## 4.8 Extensions tested and rejected

Four extra tracking features were tested with the same leave-one-game-out protocol on the
external games. Everything, including each feature's thresholds, was learned on two games and
scored on the third (`scripts/test_extensions.py`, `possession/extensions.py`):

| features | game 1 | game 2 | game 3 | mean |
|---|---|---|---|---|
| **submitted model** | 0.572 | 0.576 | 0.567 | **0.572** |
| + pressure (players within 5 m of the ball) | 0.536 | 0.519 | 0.520 | 0.525 |
| + territory (attacking third) | 0.574 | 0.559 | 0.570 | 0.568 |
| + advance (team moving forward) | 0.570 | 0.578 | 0.573 | 0.574 |
| + carrier (duration on the ball) | 0.563 | 0.561 | 0.561 | 0.561 |
| + all four | 0.530 | 0.536 | 0.537 | 0.534 |

None improves the held-out games reliably. Pressure and carrier largely repeat the ball feature,
which double-counts the same evidence and fragments possessions. Advance's +0.002 is far below the
noise of a three-game test. None was adopted.

# 5. Possession rules as implemented

* **Ball alive or dead:** dead where the physical evidence of a stoppage outweighs the cost of
  entering and leaving `D`. That evidence is walking players, a slow ball or one over a line, the
  ball lost right after going out, the early part of a set-piece tag, or a goal celebration.
  Before each kickoff tag it is dead by rule.
* **Loose and contested balls** do not change possession by themselves. A change needs the other
  team to be on or nearest the ball for long enough to pay for a transition. A deflection or a
  lost aerial duel without follow-up control is not enough, which is the challenge's "controlled
  possession" rule.
* **Restarts:** either team may restart, with equal prior. The team is decided by who is on the
  ball, the set-piece tag's team when tagged, and the phases that follow.
* **Dead-ball ownership** is not modelled, as the task asks.

# 6. Results

**External games, leave-one-game-out.** For each game, everything (bin edges, tables,
transitions, scale) is learned on the other two and tested on it:

| held-out game | spellF1 | macroF1 | scale chosen on the other two |
|---|---|---|---|
| Sample Game 1 | 0.572 | 0.789 | 6.0 |
| Sample Game 2 | 0.576 | 0.778 | 6.0 |
| Sample Game 3 | 0.567 | 0.816 | 8.0 |

**Scale curve with all three games** (mean spellF1): 0.27 at scale 2, 0.51 at 4, **0.58 at 6**,
0.57 at 8, 0.54 at 10, 0.47 at 14. It is smooth on both sides, so a moderately wrong scale costs
little. That matters, because the competition match cannot be used to check the scale.

These labels follow event-data conventions, not the competition's corrected-tracking ground
truth, so the numbers show transfer between matches; they do not predict the competition score.

**Full match.** The submitted file has 350 possessions (mean 12.2 s) and 100 stoppages, with
frames H 61,331 / A 45,004 / D 52,522 (33% of graded frames dead). Compared with the tactical
tracking alone (334 possessions, 54,028 dead frames), broadcast fusion adds 16 possessions and
reduces dead-ball frames by 1,506 (net). The dead share is within the external matches' range
(26-35%), and the possession count is slightly above it (294-343).

**Reproducibility.** The pipeline has no randomness. Two runs of `run.py` on the match data, in
different Python environments, gave the identical result. The submitted file's SHA-256 is
`6f0b718333586546f61ded5ddd58e2ca8b30307ba9975e69ec12972cb4c88a72`. Re-training on the external
games gives a byte-identical `external_params.json`.

# 7. Use of the provided data and the ground truth

* **Submitted model:** no parameter is fitted, tuned or calibrated on the competition match, its
  ground truth or its video. The match's tracking, Smart Tagging and broadcast data are inputs.
  The broadcast alignment is an automatic check using only the provided data, with thresholds fixed
  in advance.
* **Ground truth:** the submitted model was never scored on the public 5-minute window, and none
  of its parameters comes from it. Our only use of the public ground truth was early in
  development: looking at its segments to understand what counts as possession, and as a smoke
  test of early versions of the pipeline.
* **No constant is measured on the match:** every bin edge is learned on the external games, and
  the kickoff and goal features use the Smart Tagging's own tag boundaries, with no offset.
* **Fixed rules and priors** (not tuned on any data): the Smart Tagging prior tables, the quantile
  levels, the one-observation-per-second cap, the set-piece quarters, the 2 m control radius and
  the broadcast thresholds. Some were written down while working on this project. None was chosen
  by scoring against any competition data.

# 8. Training data

| dataset | used for | source and labelling | size |
|---|---|---|---|
| **Metrica Sports sample games 1-3** | all training and tuning of the submitted model | public, https://github.com/metrica-sports/sample-data. Tracking by Metrica; event data collected manually by Metrica; our possession labels derived from those events by code | 3 matches, ~430,000 frames |

The Metrica sample repository has no formal licence file; its README provides the data for the
analytics community and asks public users to acknowledge the source, which we do here. No data of
any kind from the competition match was labelled, by hand or otherwise, for training.

# 9. Limitations

* **Domain gap.** The external games come from Metrica's tracking of other matches; the
  competition tracking is automatic and noisier. The evidence cap and the smooth scale curve limit
  the damage, but frozen tables cannot adapt to this match's noise.
* **Different label conventions.** Event-derived possessions are not exactly the competition's
  ground truth (derived from corrected tracking), so the external scores do not predict ours.
* **Smart Tagging is not learned.** Its tables are fixed priors, since no external game has tags.
* **Turnovers while the ball is untracked.** Broadcast fusion recovers 4,763 frames; elsewhere,
  only slow features (tags, spread) can move possession. Ball detection in the video could target
  this, if it can be validated (4.7).
* **Geometric durations.** A semi-Markov model could encode realistic stoppage lengths.

# 10. Reproduction

```bash
# Python 3.12+ (the pinned numpy 2.5.3 requires it); verified on 3.13
pip install -r requirements.txt
# unpack the match data into data/ (see data/README.md)
python run.py                                  # ~1-2 minutes; deterministic
python evaluation/validate_submission.py submission/YamineLamal.csv
python -m pytest -q tests                      # synthetic-data tests, no match data needed
# optional: regenerate the frozen parameters from the public external data
git clone https://github.com/metrica-sports/sample-data external/metrica-sample-data
python scripts/train_external.py
```
