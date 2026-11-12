---
title: "Ball possession from automatic tracking, trained on other matches"
subtitle: "More than a Hack 2027, Phase 1: technical document"
author: "Team YamineLamal: Jerrin, Kumaran, Aditya, Sriram, Hariharan"
date: "November 2026"
---

# 1. Summary

We label every frame of FC Barcelona vs Athletic Club `H`, `A` or `D` with a **three-state hidden
Markov model** (home in possession, away in possession, dead ball) decoded by Viterbi.

**Every learned or tuned number comes from other matches.** The model is trained on ten external
matches: Metrica Sports' three public sample games and the seven matches of the DFL open data
(Bassek et al. 2025, CC BY 4.0). It is then frozen and applied unchanged to the competition
match. That covers the feature bin edges, a team-shape model, the emission probabilities, the
transition probabilities and the one evidence scale. The competition match's automatic tracking,
Smart Tagging and broadcast recording are used **as input only**. No ground truth, no video and
no human-produced data about the match are used to fit, tune or check the submitted model. This
follows the organisers' rules clarification: *"If you need to train (or retrain), you MUST use
only external data."*

Main components:

* **Labels that match the ground truth's definition.** The competition's ground truth is "`D`
  where the corrected tracking marks the ball dead, otherwise the team the tracking names"
  (CHALLENGE.md). The DFL open data carries exactly that, frame by frame, from its tracking, so
  the model learns the same notion of possession and dead ball.
* **Two-camera fusion.** The broadcast recording is aligned to the tactical timeline
  automatically, shot by shot, by matching player positions. Its ball then fills tactical frames
  where the tracker lost the ball.
* **Twelve per-frame categorical features**: nine from the tracking (including the team that
  last touched the ball, a team-shape model for frames without a ball, a still ball and the kick
  that restarts play) and three from the Smart Tagging.
* **A cap on evidence:** each feature counts as one observation per run of identical values, and
  never more than one per second, which keeps the frozen model stable on noisier tracking.
* **Speeds robust to tracking noise:** player and ball speeds are measured over half a second,
  so position jitter cannot pass for movement.
* **A benchmark on external matches** (Section 6): every design choice was kept only if it raised
  the held-out spellF1 under six kinds of simulated tracking problems (ball gaps of several kinds,
  a smoothed ball, fragmented anonymous tracks with position noise), and the final model was then
  confirmed on two DFL matches held back from all development. There it scores **0.495**
  against **0.312** for our first model (trained on Metrica only, five tracking features).

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
few parameters transfers between matches far more safely than a large one: twelve categorical
features, one 25-weight logistic model, a 3×3 transition matrix and one scale, all estimated on
ten external matches. The held-out results in Section 6 show the transfer.

# 3. Assets: used and dropped

| asset | used for | notes |
|---|---|---|
| tactical ATD `_atd.txt` + `.xml` (competition) | input: player and ball positions, which slot belongs to which team | player speeds are recomputed from positions, as the external games have no speed channel |
| Smart Tagging (competition) | input: open-play phases, set pieces, goals, kickoffs | fixed prior tables, nothing learned |
| broadcast ATD + homography (competition) | input: ball positions for tactical frames without a ball (Section 4.6) | the homography is used only to find camera cuts |
| Metrica sample games 1-3 (external) | training and tuning | https://github.com/metrica-sports/sample-data |
| DFL open data, 7 matches (external) | training, tuning and the benchmark | Bassek et al. 2025, CC BY 4.0 (Section 8) |
| tactical homography (competition) | **dropped** | a camera-zone feature was considered and dropped: no external equivalent to learn it from |
| tactical and broadcast videos | **dropped** | see 4.8 |
| public ground truth | **not used** for the submitted model | Section 7 |

**Data checks.** The tactical ATD has 199,319 contiguous records. The organisers' own
`starter/baseline.py`, run on our copy, reproduces their four published baseline files byte for
byte, which confirms the file and the `video_frame = frame_id - 1` alignment. The broadcast ATD
has 164,488 contiguous records, matching DATA.md.

# 4. Method

## 4.1 External training data and labels

**Metrica Sports' public sample games.** They are three anonymised professional matches, about 4.8 hours
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

**DFL open data.** Seven Bundesliga and 2. Bundesliga matches with optical tracking at 25 fps
(about 1,000,000 frames). On every frame the tracking provider records `BallStatus` (in play or
dead) and `BallPossession` (home or guest): the same kind of label as the competition's ground
truth, so the labels are read directly, with no derivation (`possession/dfl.py`). The DFL event
files are not used. The DFL ball is tracked almost always, unlike the competition's, so ball gaps
are added before training: a two-state chain whose probabilities are measured on the Metrica
games, separately for live and dead play (`possession/degrade.py`, profile `markov`). The DFL
labels give shorter possessions (mean 7 s) and more dead ball (41-52% of frames) than the
event-derived Metrica labels (26-35%).

## 4.2 Features (one categorical value per frame)

| feature | values | what it measures |
|---|---|---|
| `ball` | home on ball (< 2 m), home nearest, away on, away nearest, contested, or 4 bins of ball-missing stretch length | who controls the ball, and how long it has been lost |
| `pspeed` | 5 bins of mean player speed | stoppages are walked |
| `bspeed` | 4 bins of ball speed | a placed ball is slow |
| `bpos` | in, edge, out, lost right after going out | ball against the touch and goal lines |
| `spread` | home wider, even, away wider | the team in possession opens the pitch |
| `lasttouch` | home, away (or unknown) | the team that last touched the ball (below) |
| `shape` | strong home, home, even, away, strong away | a team-shape model's verdict (below) |
| `still` | still at a restart place, still in open play, moving | a placed ball |
| `restart` | kick by home, kick by away, kick with no clear taker, none | the second after a still ball is struck |
| `phase` | none, home, away, both | Smart Tagging phases mapped to "team attacking" (Build Up, Progression, Creating Chances, … name the attacker; Defending tags name the defender) |
| `setpiece` | taker (home / away / not tagged) × quarter of the tag | set-piece windows |
| `goal` | time left until the kickoff after a goal | celebration and restart |

**Bin edges are learned on the external games** as equal-share quantiles (a fixed rule set before
seeing any data). Ball-missing stretch length uses the quartiles 19 / 32 / 54 s (frame-weighted,
so long dead-ball gaps dominate). Mean player speed uses the fifths 1.0 / 1.4 / 1.9 / 2.6 m/s.
Ball speed uses the quartiles 2.7 / 5.8 / 10.5 m/s. The spread threshold is the third of |log
ratio| (0.31). The physical constants are general football facts, not tuned on any data: a player
is "on the ball" within 2 m, and the ball is out once 0.5 m over a line.

**Last touch** (`possession/touch.py`). A pass, a reception or a block changes the ball's
velocity by several m/s in a fraction of a second. With the velocity over 0.48 s before and after
each frame, a touch is a local peak of the change above 3 m/s, given to a team when one of its
players is within 2 m of the ball around the peak and at least 0.75 m closer than any opponent.
The touching team is carried forward until the next touch, a contested touch, or the ball being
missing for over a second: a loose ball stays with the team that last played it. On the training
data it agrees with the label on 97% of the live frames it covers (69% of them).

**Team shape** (`possession/shape.py`). Where the ball is not seen, the teams still show who has
it: the side in possession pushes up, spreads and moves forward. Five home-minus-away differences
(forward position of the centroid, spread in x and in y, mean speed, forward velocity), oriented
by each team's direction of attack, are taken over five time windows (1 s and 4 s centred, and
2 s shifted 1 s and 2 s ahead and 1 s back, since a team reacts late). A logistic regression
without intercept maps the 25 columns to P(home has the ball); every column changes sign when the
teams are exchanged, so the model is exactly antisymmetric and cannot carry a preference for
either side. It is fitted on the live frames of the external matches (training accuracy 85%) and
its logit is binned at fixed edges (±0.5, ±2). On frames where it is not "even" it agrees with
the label 88% of the time, and 75% where the ball is missing.

**Still ball** (`possession/deadball.py`). A ball tracked and slower than 1 m/s for at least
1 s, at a restart place (within 2 m of a line, within 2.5 m of the centre spot or a penalty spot,
or in a goal area) or elsewhere. Players walk on 49.3% of dead frames against 3.9% of live ones;
a still ball is 5.9% of dead frames against 2.7% of live ones.

**Restart kick** (`possession/deadball.py`). When a still ball (as above) is struck faster than
5 m/s within a second of the end of its still run, the following second is marked as a kick, by
the team with a player within 2 m of the ball (and 0.75 m closer than any opponent) when it was
struck. It dates the end of a stoppage and names the restarting team. On the training data,
97.5% (Metrica) and 98.9% (DFL) of marked kick frames are in the kicking team's possession. Kicks
are rare there (87 in the three Metrica games, 118 in the seven DFL matches) because the ball is
usually untracked during dead play.

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

All estimated on the ten external matches, then frozen in `models/external_params.json`:

1. **Team-shape model:** the logistic regression above, on every fifth live frame.
2. **Emission tables:** supervised counts per state from the labels, tied as above, with a mild
   Dirichlet prior. They are physically sensible: for example, the ball is missing on 91.5% of
   dead frames against 16.1% of live ones.
3. **Transitions**, measured from the labels' own spells: mean possession 8.5 s, 23% of
   possessions ending in a stoppage, mean stoppage 25.0 s.
4. **Evidence scale:** the value on a fixed grid with the best mean spellF1 over the ten matches,
   computed with the organisers' own `possession_metrics.py`: **8.0**.

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

**Runtime.** The alignment is the slow part of the pipeline: about a minute on a 4- to 8-core
laptop, against about 10 seconds for everything else (reading the tactical tracking, the features
and the decoding). Each shot's offset is searched over every frame of the match rather than near
a guess, because a shot can show any moment (replays included) and a faster screening step we
tried overstated the margin between the best and second-best offsets, which the acceptance test
relies on. The search runs in parallel on all cores. `--no-broadcast` skips the fusion and runs
in about 10 seconds, but produces a different file; the submitted file uses the fusion.

## 4.7 Handling missing and noisy tracking

The competition tracking is automatic and uncorrected: it has gaps, fragmented tracks and identity
switches, and the ball is missing on 41% of graded frames (DATA.md). Each problem is handled as
follows.

| problem | how it is handled |
|---|---|
| **ball missing** | where an aligned broadcast frame has the ball, it fills the gap (4.6: 4,763 frames); an observed tactical ball is never overridden. Otherwise the `ball` feature records how long the ball has been missing, in four bins learned on the external games (19 / 32 / 54 s), so a long gap is evidence of a stoppage while a short one is not. The training data has comparable missingness: the Metrica games track the ball on 59-68% of graded frames, and the DFL matches get gaps simulated from the Metrica tracking |
| **possession while the ball is missing** | the `lasttouch` feature carries the team that last played the ball through short gaps, and the team-shape model gives evidence on frames without a ball (75% agreement there on the training data); a change of state also costs a transition, so without contrary evidence the last team keeps possession |
| **players missing** | features are built from whoever is tracked. A quantity that cannot be measured reliably becomes a **neutral** value, with a likelihood ratio of 1, never a guess: mean speed when fewer than 8 players are tracked, team spread when a team has fewer than 5, ball speed, ball position and still ball when the ball is missing (except "lost right after going out", which is evidence), team shape when a team has fewer than 5 players tracked, last touch after a second without a ball |
| **position noise** | speeds are measured over a chord of ±6 frames (0.48 s) rather than frame to frame, so random jitter cancels out instead of adding up: with 10 cm of noise, frame-to-frame speeds read about 4.6 m/s for standing players, which erased the "players walk during stoppages" evidence in our tests. The benchmark's `full` profile adds this noise (Section 6) |
| **identity switches** | no feature follows an individual player. They use which **team** is nearest the ball and team-level aggregates (the team-shape model included), so a track slot changing player does not matter. The one per-slot quantity, speed from consecutive positions, is smoothed over ±0.2 s per player, averaged over all tracked players and smoothed again over ±0.5 s, which limits a jump in one slot to a brief, diluted blip; a speed above 12 m/s (faster than any player) is a track jumping to another object and is dropped. In the benchmark, cutting every track into anonymous fragments over 46 slots changed no score |
| **duplicated objects** (more than 22 tracked) | nearest-player distances are unaffected; aggregates change marginally |
| **smoothed ball** | the competition's automatic ball track is smoothed over about a second, which blurs touches. Touches are measured over 0.48 s chords, and the benchmark includes a profile with the ball smoothed over 1 s (Section 6), where the model keeps its advantage |
| **noise and flicker** | each feature counts as one observation per run of identical values, capped at **one per second** (4.3), and Viterbi requires enough evidence to pay for two transitions before it splits a possession. This strongly limits how far jittery tracking can fragment possessions; without the cap, the model's quality collapsed just above its best scale (4.3), and with it the curve is smooth (Section 6) |

## 4.8 Considered and not used: ball detection in the video

We also built a ball detector for the video, using a public model pre-trained on external images.
It could not be validated on the external games, which have no video. We kept only components
whose benefit we could measure without the competition ground truth, so it is not part of the
submission or the repository.

## 4.9 Features tested and rejected

Before the benchmark of Section 6 existed, with the first model (Metrica only), four extra
tracking features were tested with a leave-one-game-out protocol on the three Metrica games. Everything, including each feature's thresholds, was learned on two games and
scored on the third (`scripts/test_extensions.py`, `possession/extensions.py`):

| features | game 1 | game 2 | game 3 | mean |
|---|---|---|---|---|
| **first model** | 0.572 | 0.576 | 0.567 | **0.572** |
| + pressure (players within 5 m of the ball) | 0.536 | 0.519 | 0.520 | 0.525 |
| + territory (attacking third) | 0.574 | 0.559 | 0.570 | 0.568 |
| + advance (team moving forward) | 0.570 | 0.578 | 0.573 | 0.574 |
| + carrier (duration on the ball) | 0.563 | 0.561 | 0.561 | 0.561 |
| + all four | 0.530 | 0.536 | 0.537 | 0.534 |

None improved the held-out games reliably. Pressure and carrier largely repeat the ball feature,
which double-counts the same evidence and fragments possessions. Advance's +0.002 is far below the
noise of a three-game test. None was adopted.

We also tested a stronger learned model under the same protocol: a gradient-boosted classifier
(LightGBM) on 14 raw tracking measurements, made home/away symmetric by mirroring, decoded with
the same Viterbi, with its strength chosen by cross-fitting within the two training games. It
scored 0.555 / 0.628 / 0.583 (mean 0.588 against 0.572): better on two games, worse on one, with
its strength setting changing between folds (0.10 and 0.20). The +0.017 mean is within the noise
of a three-game test, and the classifier has no cap on evidence for tracking noisier than its
training data, so it was not adopted.

On the benchmark of Section 6, two more were rejected: the momentary touch itself (as opposed
to the last touch: no gain once `lasttouch` is in), and "gap context" (where the ball was lost and
where it reappears), which separates dead from live play well (57% of dead frames are "lost at a
line, back at a restart place" against 6% of live ones) but adds nothing measurable once `still`
and the existing ball features are in (+0.0005).

# 5. Possession rules as implemented

* **Ball alive or dead:** dead where the physical evidence of a stoppage outweighs the cost of
  entering and leaving `D`. That evidence is walking players, a slow or still ball (more so at a
  restart place) or one over a line, the ball lost right after going out, a long ball gap, the
  early part of a set-piece tag, or a goal celebration. A still ball being struck marks the
  restart.
  Before each kickoff tag it is dead by rule.
* **Loose and contested balls** do not change possession by themselves: the last team to touch
  the ball keeps it, and a contested touch clears the last-touch evidence. A change needs the other
  team to touch or be on the ball, with enough evidence to pay for a transition. A deflection or a
  lost aerial duel without follow-up control is not enough, which is the challenge's "controlled
  possession" rule.
* **Restarts:** either team may restart, with equal prior. The team is decided by who is on the
  ball, the set-piece tag's team when tagged, and the phases that follow.
* **Dead-ball ownership** is not modelled, as the task asks.

# 6. Results

**The benchmark** (`scripts/benchmark.py`, external matches only). The decision rule was written
down before the runs: a change is kept when its mean spellF1 on the DFL development matches is
higher (their labels are made the way the ground truth is), its Metrica mean is not more than
0.03 lower, and the final check does not reverse the gain.

* **Development:** the three Metrica games and five DFL matches. For each one, everything (bin
  edges, team-shape model, tables, transitions, scale) is learned on the other development
  matches and the held-out one is decoded and scored with the organisers' spellF1. Metrica games
  are scored with their own tracking gaps. DFL matches are scored under six profiles
  (`possession/degrade.py`), each with three seeds: `markov` (gaps as on the Metrica tracking),
  `short` (mean 0.5 s), `long` (4 s), `crowded` (2 s, starting where players crowd the ball),
  each removing 41% of the ball as DATA.md reports, `smooth` (markov gaps, ball smoothed
  over 1 s, as the competition's ball is), and `full` (crowded gaps, every player track cut into
  fragments of mean 60 s in 23 anonymous slots per team as in the competition's 46, and Gaussian
  position noise of 10 cm for players and 20 cm for the ball). `full` was added last; the table
  below of the development steps was run before it existed.
* **Final check:** two DFL matches (J03WQQ and J03WR9, the last two by id) were held back from all
  development and scored once, with the model trained on every development match.

Development steps (five profiles, before `full` was added):

| model | Metrica dev | DFL dev | DFL reserved |
|---|---|---|---|
| A: first model (Metrica only, five tracking features) | **0.572** | 0.286 | 0.320 |
| B: A trained on Metrica + DFL | 0.556 | 0.328 | 0.371 |
| F: B + last touch + team shape | 0.668 | 0.442 | 0.483 |
| I: F + still ball | 0.650 | 0.458 | 0.500 |
| K: I + restart kick | 0.650 | 0.460 | 0.501 |

The `full` profile then showed one weakness of K: position noise made every player look fast
(4.6 m/s median for standing players), and spellF1 under `full` fell to 0.346 against 0.44-0.52
under the other profiles. Measuring speeds over a chord (4.7) fixed it. Chords of ±3 and ±6
frames were compared, and ±6 had the higher DFL mean. With all six profiles:

| model | Metrica dev | DFL dev | DFL dev macroF1 | DFL reserved |
|---|---|---|---|---|
| A: first model | 0.572 | 0.278 | 0.674 | 0.312 |
| K: frame-to-frame speeds | 0.650 | 0.441 | 0.772 | 0.482 |
| **L: K with speeds over ±6 frames (submitted)** | **0.658** | **0.455** | **0.804** | **0.495** |

Under `full`, L scores 0.455 against 0.346 for K; under the other profiles it is within 0.011 of
K (lower by 0.001-0.010). It is higher than K on all five DFL development matches, on all three
Metrica games and on both reserved matches (0.503 and 0.487 against 0.495 and 0.469).

**A second held-out split.** Retraining on Metrica and the other five DFL matches and holding out
J03WN1 and J03WR9 instead gives the same picture: L scores 0.432 and 0.491 (all six profiles),
against 0.418 and 0.474 for K, and 0.436 and 0.492 under `full`.

Mean spellF1; DFL figures are means over matches, profiles and seeds. Steps B, F and I each raised
the DFL mean on every development match; the restart kick (K) is a small gain (+0.003, better
under all five profiles and on four of five matches, Metrica unchanged). The submitted model (L)
is better than A on all five DFL development matches (0.430-0.472 against 0.232-0.303), under
every profile (markov 0.514, short 0.445, long 0.460, crowded 0.433, smooth 0.421, full 0.455,
against 0.337 / 0.284 / 0.254 / 0.226 / 0.326 / 0.241), and on both reserved matches (0.503 and
0.487 against 0.320 and 0.304; macroF1 0.802 and 0.824 against 0.666 and 0.704). On Metrica it is
0.647 / 0.643 / 0.684 against 0.572 / 0.576 / 0.567.

**Before the reserved matches were set aside,** two earlier explorations had already scored all
seven DFL matches: training on Metrica + DFL (better on 7 of 7) and the "gap context" feature (no
gain). The reserved matches were therefore not untouched by every decision, but none of the
features of models F and I had been run on them before the final check. They were used again,
for the restart kick (a tie, +0.0003, which the rule accepts as "not reversed") and for the speed
chord (+0.008 and +0.018), each time after the change had passed on the development matches.
Repeated looks weaken them as an untouched test; the second split above is a further check.

**Scale curve with all ten matches** (mean spellF1): 0.40 at scale 2, 0.55 at 4, 0.564 at 6,
**0.569 at 8**, 0.561 at 10, 0.524 at 16. It is flat around the optimum, so a moderately wrong
scale costs little.

These scores show transfer between matches; they do not predict the competition score, which
depends on that match's tracking and its own ground truth.

**Full match.** `run.py` prints the output's possession and stoppage counts and a set of
label-free checks (ball-missing shares in live and dead output, dead share per half, coverage of
the last-touch and team-shape features), stored in `submission/YamineLamal_report.json`.

**Reproducibility.** The pipeline has no randomness. Two runs of `run.py` on the same data give
the identical file. Re-training on the external matches gives the same `external_params.json` on
the same platform.

# 7. Use of the provided data and the ground truth

* **Submitted model:** no parameter is fitted, tuned or calibrated on the competition match, its
  ground truth or its video. The match's tracking, Smart Tagging and broadcast data are inputs.
  The broadcast alignment is an automatic check using only the provided data, with thresholds fixed
  in advance.
* **Ground truth:** the submitted model was never scored on the public 5-minute window, and none
  of its parameters comes from it. Our only use of the public ground truth was early in
  development: looking at its segments to understand what counts as possession, and as a smoke
  test of early versions of the pipeline.
* **No constant is measured on the match:** every bin edge and the team-shape model are learned
  on the external matches, and the kickoff and goal features use the Smart Tagging's own tag
  boundaries, with no offset. The only figure taken from the organisers' documents is the 41%
  ball-missing share (DATA.md) used to simulate gaps in the external benchmark.
* **Fixed rules and priors** (not tuned on any data): the Smart Tagging prior tables, the quantile
  levels, the one-observation-per-second cap, the set-piece quarters, the 2 m control radius and
  the broadcast thresholds, the touch, still-ball and kick thresholds, the restart places and the
maximum player and ball speeds. The speed chord (±6 frames) was chosen on the external benchmark. Some were
written down while working on this project. None was chosen
  by scoring against any competition data.

# 8. Training data

| dataset | used for | source and labelling | size |
|---|---|---|---|
| **Metrica Sports sample games 1-3** | training and tuning | public, https://github.com/metrica-sports/sample-data. Tracking by Metrica; event data collected manually by Metrica; our possession labels derived from those events by code | 3 matches, ~430,000 frames |
| **DFL open data (IDSSE)** | training, tuning and the benchmark | Bassek, M., Rein, R., Weber, H., Memmert, D. (2025). An integrated dataset of spatiotemporal and event data in elite soccer. *Scientific Data* 12, 195. doi:10.1038/s41597-025-04505-y. Data © Deutsche Fußball Liga, CC BY 4.0, https://doi.org/10.6084/m9.figshare.28196177. Labels are the tracking's own ball status and possession; the event files are not used | 7 matches, ~1,000,000 frames |

The Metrica sample repository has no formal licence file; its README provides the data for the
analytics community and asks public users to acknowledge the source, which we do here. No data of
any kind from the competition match was labelled, by hand or otherwise, for training. Neither
dataset is redistributed: `scripts/download_dfl.py` and the Metrica clone fetch them from the
source.

# 9. Limitations

* **Domain gap.** The external matches come from other trackers and leagues; the competition
  tracking is automatic, fragmented and noisier. The DFL gaps are simulated, not real tracker
  failures. The evidence cap, the six benchmark profiles (including fragmented, noisy tracks), the
  noise-robust speeds and the flat scale curve limit the risk, but
  frozen tables cannot adapt to this match's noise.
* **Label conventions.** DFL's tracking-based labels follow the same definition as the ground
  truth, but each provider applies it its own way, and the Metrica labels are event-derived. The
  external scores therefore show transfer; they do not predict ours.
* **A small final check.** Two reserved matches are strong evidence of direction (the gain held on
  both, under every profile, and on a second held-out split), not a precise estimate of its size.
  They were used for more than one decision (Section 6), which weakens them as an untouched test.
* **Smart Tagging is not learned.** Its tables are fixed priors, since no external game has tags.
* **Turnovers while the ball is untracked.** Broadcast fusion recovers 4,763 frames; elsewhere,
  the team-shape model and the slower features have to move possession, and the shape model is
  right less often without the ball (75%) than with it. Ball detection in the video could target
  this, if it can be validated (4.8).
* **Durations.** The model treats possession and stoppage durations as geometric. Measured on the
  external matches' labels, that is close to reality: possession lengths vary *more* than
  geometric (coefficient of variation 1.16 on Metrica and 1.37 on DFL, with 16% and 30% of
  possessions shorter than 2 s), and stoppages vary about as much (0.92 and 0.88). A semi-Markov model, which would favour typical lengths, is therefore not
  expected to help.

# 10. Reproduction

```bash
# Python 3.12+ (the pinned numpy 2.5.3 requires it); verified on 3.13
pip install -r requirements.txt
# unpack the match data into data/ (see data/README.md)
python run.py                                  # ~1 minute (most of it the broadcast alignment); deterministic
python evaluation/validate_submission.py submission/YamineLamal.csv
python -m pytest -q tests                      # synthetic-data tests, no match data needed
# optional: the same in a pinned container (Dockerfile)
docker build -t yaminelamal .
docker run --rm -v "$PWD/data:/app/data:ro" -v "$PWD/submission:/app/submission" yaminelamal
# optional: re-train and re-run the benchmark from the public external data
git clone https://github.com/metrica-sports/sample-data external/metrica-sample-data
python scripts/download_dfl.py                 # DFL open data, ~2.6 GB
python scripts/train_external.py               # rewrites models/external_params.json
python scripts/benchmark.py                    # held-out results of Section 6
```
