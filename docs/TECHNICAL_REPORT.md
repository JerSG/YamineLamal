---
title: "Ball possession from automatic tracking: a self-calibrating hidden Markov model"
subtitle: "More than a Hack 2027, Phase 1: technical document"
author: "TEAM_NAME"
date: "November 2026"
---

# 1. Summary

We label every frame of FC Barcelona vs Athletic Club `H`, `A` or `D` with a **three-state hidden
Markov model** (home in possession, away in possession, dead ball):

* **Nine per-frame categorical features**, built from the automatic tracking, the Smart Tagging
  and the tactical homography.
* **Emission tables learned by unsupervised EM over the whole match.** No label of any kind is
  read. The home and away states are tied by a mirror symmetry, and the dead-ball state may only
  be learned from features that physically measure a stoppage.
* **Per-feature tempering**: each feature counts once per *independent look*, measured from the
  data as the mean run length of its own codes.
* **A self-consistent evidence scale**: one global multiplier, set so that the decoded
  possessions have the mean length the transition model itself assumes.
* **Viterbi decoding** per half, because the ranking metric scores whole possessions and Viterbi
  returns the most likely *segmentation*.

| on the public 5 minutes (sanity check only) | macroF1 | spellF1 | possessions |
|---|---|---|---|
| ground truth | | | 17 |
| organisers' floor, `baseline_smooth50` | 0.621 | 0.326 | 26 |
| organisers' `baseline_smooth125` | 0.581 | 0.424 | 16 |
| **this submission** | **0.668** | **0.423** | **18** |

On these 17 possessions the bootstrap 5-95% interval of our spellF1 is 0.27 to 0.58, so the
window cannot separate us from `baseline_smooth125`. The repository shows that
`baseline_smooth125` *loses* to `smooth50` on the full match (0.224 vs 0.243). Our macroF1 is
the best of the five reference files, and our possession count (18 vs 17) is the closest.

The public window holds 17 possessions, and the repository shows that it ranks two baselines in
the wrong order. We report it because it is the only external number available. **No parameter
was chosen by scoring against it** (Section 7).

# 2. Why this approach

The metric decides the design. `spellF1` matches *possessions* one-to-one by temporal IoU, so the
expensive mistakes are a possession split in two by a flicker, two possessions merged across a
turnover, and a stoppage that is missed. All three are segmentation errors. A per-frame classifier
followed by smoothing, which is the baseline, has no notion of a segment. An HMM decoded with
Viterbi does: it trades each frame's evidence against the cost of a transition, so a short burst
of contrary evidence cannot split a possession unless it is strong enough to pay for two
transitions.

We chose an **unsupervised** model because the rules leave no labelled data for this game, and
because we did not have labelled tracking from other games of this provider. EM can learn how
much each feature is worth from the match itself, for example how often the nearest player
belongs to the team without the ball, provided the model is structured so that its states keep
their meaning (Section 4.3).

# 3. Assets: used and dropped

| asset | used for | notes |
|---|---|---|
| tactical ATD, `_atd.txt` + `.xml` | ball and player positions, player speed, team of each track slot, attack direction | **Speeds are m/s**, although the metadata labels them `kph`: median 1.3, 99.9th percentile 8.1. The ball's own speed channel is almost always NaN, so ball speed is computed from positions. |
| Smart Tagging | open-play phases (which team attacks), set pieces, goals, kickoffs | `N/A` team on every throw-in and kickoff, handled as its own value |
| tactical homography | where the camera points (play location on every frame, including the 41% without a ball) | frames with `valid = 0` or reprojection error > 5 are "unknown" |
| broadcast video, ATD, homography | **dropped** | not synchronised to the tactical timeline. Aligning it is a project of its own, and it covers fewer frames |
| tactical video | **dropped** | we used no video decoding. Everything comes from the provided tracking |
| `evaluation/results/baseline_raw.csv` | only in `--mode basic`, a fallback when the raw ATD is absent | the organisers' unsmoothed nearest-player signal, computed from the ATD alone. In full mode we recompute it from the ATD and check that it agrees on **100.00%** of frames |

**Data checks we ran.** The ATD has 199,319 records numbered 1..199,319, contiguous, every line in
the declared `frame:players:ball` layout. The organisers' `starter/baseline.py`, run on our copy
of the ATD, reproduces all four committed baseline files byte for byte. That confirms both the
file and the `video_frame = frame_id - 1` alignment.

# 4. Method

## 4.1 Features (one categorical value per frame)

| feature | values | source | what it measures |
|---|---|---|---|
| `ball` | home_on, home_near, away_on, away_near, contest, gap<1s, gap1-4s, gap4-10s, gap>10s | ATD | team of the nearest tracked player, split by whether they are within 2 m of the ball, "contest" if both teams are. When the ball is missing: the length of the missing stretch |
| `pspeed` | 5 bins of mean player speed (0.9 / 1.3 / 1.8 / 2.5 m/s), unknown | ATD | stoppages are walked |
| `bspeed` | still, rolling, passed, struck, unknown | ATD | a placed ball is still |
| `bpos` | in, edge, out, miss_after_out, miss | ATD | ball against the touch and goal lines (0.5 m margin), and "lost straight after being seen over a line" |
| `spread` | home_wider, even, away_wider, unknown | ATD | log ratio of the two teams' spatial spread. The team in possession opens the pitch |
| `phase` | none, home, away, both | Smart Tagging | open-play phases mapped to "team attacking": Build Up, Progression, Creating Chances, Attacking Transition, Ball in Final Third / Box, Players in the Box and Long Ball name the attacker. The three Defending tags and Defensive Transition name the defender, so the other team is the attacker |
| `setpiece` | none, {home, away, n/a} × {0-3, 3-8, 8-15, 15+ s into the tag} | Smart Tagging | non-kickoff set pieces are fixed 35 s windows. EM learns where inside them the dead ball turns live |
| `goal` | none, 4 bins of time left until the restart kickoff | Smart Tagging | celebration and restart after a goal |
| `camzone` | home_att, mid, away_att, unknown | homography | third of the pitch at the image centre, relative to each team's attack direction (from the ATD metadata, swapped at half time) |

Team-bearing features are written in absolute terms (home/away). The model mirrors them, see 4.3.

**Hard constraint (kickoffs).** Kickoff tags are fixed 5 s windows. The two published kickoff
frames (11,185 and 113,742, DATA.md) sit 88 and 82 frames after their tags' starts, so we place
the kick 85 frames in. Frames of a half before its kickoff, and frames inside any kickoff tag
before the kick (including kickoffs after goals), can only be `D`. This covers the 2,934 graded
dead frames before the second-half kickoff that FAQ.md warns about. It is derived from the tags,
not hard-coded to frames.

## 4.2 Model

States `H`, `A`, `D`. The transition matrix is **fixed** from football priors, not learned:

* a possession spell lasts **12 s** on average once control is required (touches and deflections
  do not end it);
* **a third** of spells end in a stoppage, the rest in an open-play turnover;
* a stoppage lasts **18 s** on average;
* a restart goes to either team with equal probability.

Emissions are a naive-Bayes product over features. The per-frame log-likelihood of state *s* is

    log p(o_t | s) = scale · Σ_f  w_f · log B_f[s, o_{f,t}]

## 4.3 Keeping the states meaningful without labels

Our first, unconstrained version of EM converged to a model in which "D" meant "no Smart Tagging
phase is running" (P = 0.987) and, in a later version, "ball not tracked". Both were good
clusters and bad dead balls. Three structural rules fixed this, and none of them looks at labels:

1. **Home/away mirror tie.** `H` and `A` share one emission table, with values swapped
   home↔away. A feature cannot mean "Barcelona"; it can only mean "the team in possession" or
   "the other team". The `D` table is mirror-symmetric, since a dead ball favours neither team.
   A unit test checks that swapping the teams in the input swaps `H` and `A` in the output,
   frame for frame.
2. **Only physical evidence may define D.** The dead-ball table is learned only for `pspeed`,
   `bspeed` and `bpos`. For every other feature it stays at a fixed, mild prior. The camera zone
   and goal timing are slow context signals, so they keep fixed tables for both states.
3. **Neutral values.** "Not measurable on this frame" values (`unknown`, a plain `miss`) carry a
   likelihood ratio of 1. Missingness is modelled once, by the `ball` feature, instead of three
   times.

The learned tables are physically sensible. On dead-ball frames the mean player speed is below
0.9 m/s 36.8% of the time, against 4.0% on live frames. "Ball lost straight
after going out" is 59.8% vs 0.1%. A still ball is 23.4% vs
5.8%. The tables are dumped in `submission/<team>_report.json`.

## 4.4 How much each feature counts

Consecutive frames are not independent looks: a Smart Tagging phase that lasts 20 s says one
thing, not 500 things. Each feature *f* gets the weight `w_f = min(0.2, 1 / r_f)`, where `r_f` is
the mean run length (in frames) of that feature's codes over the match. With a weight equal to
the reciprocal of its run length, a feature contributes roughly one observation per change of
value.

| feature | ball | pspeed | bspeed | bpos | spread | phase | setpiece | goal | camzone |
|---|---|---|---|---|---|---|---|---|---|
| mean run length = one look per … frames | 31 | 71 | 42 | 256 | 402 | 609 | 625 | 7565 | 749 |

## 4.5 The evidence scale: self-consistency instead of tuning

The relative weights fix how features trade off against each other. One number is still free:
the total weight of evidence against transitions, which sets how many possessions come out. The
baselines show that choosing it on the public window picks the wrong value. We choose it by
**self-consistency**. The transition model says a spell lasts 12 s on average, so the evidence
scale is set, by bisection, to the value at which the *decoded* spells also average 12 s. EM and
calibration alternate until the scale moves by less than 3% (it settles at 3.32 after
7 rounds). The only input is the 12 s football prior. No label is read.

## 4.6 Decoding and output

Viterbi runs separately on each graded half. Graded frames get `H`, `A` or `D`. Ungraded frames
(before kickoff, half time, after the final whistle) get `X`, which is never scored. We never
abstain on a graded frame: under spellF1 an `X` repeats the previous label anyway, and under
macroF1 it is a guaranteed miss.

# 5. Possession rules as implemented

* **Ball alive or dead.** Dead where the evidence of a stoppage (walking players, ball still or
  over a line, ball lost right after going out, early in a set-piece window, a goal celebration)
  outweighs the cost of entering and leaving `D`. Before every kickoff it is dead by constraint.
* **Loose balls and contested balls** do not change possession by themselves. A change needs the
  other team to be on or nearest the ball long enough to pay for a transition, roughly a second
  or more of consistent evidence at the calibrated scale. A deflection or a lost aerial duel
  without follow-up control produces too little evidence. This matches the challenge's
  "controlled possession" rule.
* **Restarts.** After a stoppage either team may restart (equal prior). The team is decided by
  who is on the ball, which side the set-piece tag names (when it names one), and the phases that
  follow.
* **Dead-ball ownership** is not modelled, as the task asks.

# 6. Results and robustness

**Full match (no ground truth available to us):** 300 possessions of mean
12.0 s, 114 stoppages, frames H 53,494 / A 36,201 / D 69,162
(43.5% dead). In professional football the ball is in play for about 55 to 60 of ~95
minutes, so a dead share of around 40% of graded frames is realistic.

**Public 5 minutes:** macroF1 0.668, spellF1 0.423, 18 possessions vs 17
true, alive/dead F1 0.893, team accuracy on frames both call alive 0.891.

**Sensitivity (unsupervised).** `scripts/sensitivity.py` changes one prior at a time and compares
the full-match output with the reference run. No ground truth is involved.

| variant | frames unchanged | possessions | mean spell (s) |
|---|---|---|---|
| reference | 100.00% | 300 | 12.0 |
| alive spell 10 s | 98.52% | 361 | 10.0 |
| alive spell 15 s | 97.83% | 238 | 14.9 |
| stoppage share 0.25 | 99.95% | 298 | 12.0 |
| stoppage share 0.40 | 99.95% | 298 | 12.0 |
| dead spell 12 s | 100.00% | 300 | 12.0 |
| dead spell 25 s | 99.95% | 298 | 12.0 |
| prior strength 500 | 99.57% | 301 | 12.0 |
| prior strength 8000 | 99.58% | 298 | 12.0 |
| control radius 1.5 m | 98.74% | 297 | 12.1 |
| control radius 3 m | 96.62% | 294 | 12.0 |
| no camera | 99.93% | 298 | 12.0 |

Every perturbation leaves at least 96.6% of graded frames unchanged. The only prior that moves the possession count is the assumed mean spell length, because it is the target of the self-consistent scale (4.5). Even then the frame labels barely move: the count changes because short spells split or merge at the margin. The camera feature contributes little (99.93% unchanged without it). We keep it because it is the only location signal on frames without a ball.

# 7. Use of the public ground truth

The public 5 minutes were used to (1) confirm the file format and frame alignment, and (2) as a
smoke test while debugging. Each structural fix in 4.3 was triggered by inspecting the *learned
tables*, which were visibly wrong (D meaning "no tag", walking-speed bins meaning nothing), and
the public score was then checked to confirm the pipeline was no longer broken. No hyperparameter
was searched against it. Every constant in `possession/config.py` is tagged with its source
(`[football]`, `[data]` or `[model]`), and the one free scalar is set by self-consistency
(4.5). The kickoff offset comes from the two kickoff frames published in DATA.md, which are
boundaries rather than labels.

# 8. Training data

**None.** No labelled dataset of any kind was used, from this game or any other. All parameters
are either football priors written down before seeing labels, or learned by unsupervised EM on the
provided tracking and tagging of this game.

# 9. Limitations and what we would do next

* **Turnovers while the ball is not tracked.** Without a ball, possession is carried forward, and
  only slow features (tags, spread, camera zone) can move it. A turnover inside a long gap is
  found late or not at all. Next step: a learned "ball location" model from player motion
  (players converge on the ball), or ball detection on the tactical video.
* **Geometric durations.** A plain HMM favours very short and very long stoppages equally per
  frame. An explicit-duration (semi-Markov) model would encode that throw-ins last ~10 s and goal
  celebrations ~60 s.
* **Naive Bayes.** Features are treated as independent given the state. Run-length tempering
  corrects the worst double counting, but the `ball` and `bspeed` features remain correlated.
* **The 12 s prior sets the possession count.** It is a football prior, not a measurement of this
  match. The sensitivity table shows the frame labels barely move, while the count follows the
  prior by construction.
* **Team-symmetric model.** Barcelona's and Athletic's styles differ (spread, pressing height).
  The mirror tie cannot represent that. It is the price of identifiability without labels.

# 10. Reproduction

```bash
pip install -r requirements.txt               # numpy, pandas (scorer), pytest (tests)
# unpack the match data into data/ (see data/README.md)
python run.py --team TEAM_NAME                # ~1 minute; deterministic
python evaluation/validate_submission.py submission/TEAM_NAME.csv
python -m pytest -q tests                     # synthetic-data tests, no match data needed
python scripts/sensitivity.py                 # optional: the robustness table above (~40 min)
```

Verified: two clean runs of `python run.py` give byte-identical files (SHA-256
`f8e92b00bd163ad0803906229bebb6db25b51143e8ee451e6fc966f99e8b4a3d`).
