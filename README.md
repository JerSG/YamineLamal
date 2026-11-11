# Ball possession from automatic tracking, trained on other matches (team YamineLamal)

<img alt="Animated, simulated passage of play: 22 players and the ball, with the possession label switching from home to away to dead ball and back." src="docs/assets/pitch.svg" width="100%">

<sub>Simulated players, not match data (drawn by <code>scripts/make_banner.py</code>).</sub>

Team **YamineLamal** (Jerrin, Kumaran, Aditya, Sriram, Hariharan) submission for
**More than a Hack 2027, Phase 1: Ball Possession** (Metrica Sports · FC Barcelona · Mobile World
Capital Barcelona).

For every frame of FC Barcelona vs Athletic Club, the pipeline outputs `H`, `A` or `D` with a
three-state hidden Markov model decoded by Viterbi. **Every learned or tuned number comes from
other matches**: Metrica's three public sample games and the seven matches of the DFL open data,
whose labels are made the way the competition's ground truth is (ball status and possession from
the tracking). The model is trained there, frozen, and applied unchanged to the competition match. It reads the
competition match's automatic tracking, Smart Tagging and broadcast recording **as input only**.
No ground truth, no video and no hand labels of the match are used for anything, and no parameter
is fitted or tuned on the match. See [`docs/TECHNICAL_REPORT.md`](docs/TECHNICAL_REPORT.md).

```
Metrica sample games + DFL open data ─> train_external.py ─> models/external_params.json  (frozen)
                                                                       │
tactical tracking ──────────────────────────────┐                      ▼
broadcast tracking ─> align shots ─> fill ball ─┼─> 12 features ─> frozen HMM ─> Viterbi ─> H/A/D
Smart Tagging ──────────────────────────────────┘
```

On ten external matches, every change was kept only if it raised the held-out spellF1. On two DFL
matches held back until the end, the submitted model scores **0.495** against **0.312** for our
first model ([report, Section 6](docs/TECHNICAL_REPORT.md)).

## Reproduce the submission

**Python 3.12 or newer** (verified on 3.13; the pinned numpy 2.5.3 needs 3.12+).

```bash
pip install -r requirements.txt
```

Unpack the match data into `data/` with the names it ships with ([`data/README.md`](data/README.md)).
Six files are read; no video is needed:

```
data/tactical/BAR-ATH_tactical_atd.xml
data/tactical/BAR-ATH_tactical_atd.txt
data/tactical/BAR-ATH_tactical_smart_tagging.xml
data/broadcast/BAR-ATH_broadcast_atd.xml
data/broadcast/BAR-ATH_broadcast_atd.txt
data/broadcast/BAR-ATH_broadcast_homography.csv
```

```bash
python run.py
python evaluation/validate_submission.py submission/YamineLamal.csv
```

`run.py` takes about 1-2 minutes on a 4-core laptop and writes `submission/YamineLamal.csv` and
`submission/YamineLamal_report.json`, then writes the replay page `submission/viewer.html` and
opens it in the browser (see below). It also prints label-free checks (aggregates only, such as
the dead-ball share and how often the ball is missing in live and dead output). It is
deterministic: the same data and command give a byte-identical file. The frozen parameters it
uses are committed in `models/external_params.json`.

With Docker instead (not needed; the same pinned versions):

```bash
docker build -t yaminelamal .
docker run --rm -v "$PWD/data:/app/data:ro" -v "$PWD/submission:/app/submission" yaminelamal
```

### Watch the output

`python run.py` builds the replay page automatically, after the possession file is written (the
page never changes that file). To rebuild it on its own, or to view a public sample game:

```bash
python scripts/view_match.py              # needs data/ and submission/YamineLamal.csv
python scripts/view_match.py --external 1 # a public Metrica sample game, no match data needed
```

The page is one self-contained HTML file: a 2D replay of the tracking
with players coloured by team, the ball (drawn hollow where it was filled from the broadcast),
the current label, and a full-match timeline of the labels that you can click to seek. The
`--external` view adds a second timeline with that game's event-derived labels. The page embeds
tracking, so it is written to `submission/` (gitignored) and is for local use only. It shows the
output and is not a labelling tool: nothing seen in it goes back into the model, and no ground
truth is read.

### Re-training and the benchmark (optional)

The parameters can be regenerated from the public external data:

```bash
git clone https://github.com/metrica-sports/sample-data external/metrica-sample-data
python scripts/download_dfl.py       # DFL open data, ~2.6 GB, into external/dfl
python scripts/train_external.py     # ~1 minute; rewrites models/external_params.json
python scripts/benchmark.py          # held-out results on the external matches (~10 minutes)
```

None of them reads competition data.

`run.py` has no randomness: on the official data bundle it reproduces the submitted file byte for
byte. To compare two runs, `Get-FileHash submission\YamineLamal.csv` (Windows) or
`sha256sum submission/YamineLamal.csv` gives the file's SHA-256.

### Options

| flag | effect |
|---|---|
| `--no-broadcast` | skip broadcast fusion (tactical tracking only) |
| `--no-viewer` | do not write the replay page |
| `--no-open` | write the replay page but do not open the browser |

## How it is trained (external data only)

* **Data:** Metrica Sports' three public sample games (https://github.com/metrica-sports/sample-data),
  with labels derived from their event data (`possession/external.py`), and the seven matches of
  the DFL open data (Bassek et al. 2025, CC BY 4.0), whose labels are the tracking's own ball
  status and possession, the same definition as the competition's ground truth
  (`possession/dfl.py`). The DFL ball is almost always tracked, so realistic gaps are added
  (`possession/degrade.py`).
* **Features:** who is on or nearest the ball and for how long it has been missing, player and
  ball speed, ball against the lines, team spread, **the team that last touched the ball**
  (`possession/touch.py`), **a team-shape model** for frames without a ball (`possession/shape.py`,
  a home/away-antisymmetric logistic regression), **a still ball at a restart place**
  and **the kick that restarts play** (`possession/deadball.py`), and three Smart Tagging features.
* **Learned there, then frozen:** the feature bin edges (equal-share quantiles), the team-shape
  model, the emission tables, the transition probabilities and the evidence scale (chosen to
  maximise the organisers' own spellF1 on those matches).
* **Fixed rules, no data:** each run of identical feature values is one observation, capped at
  one per second. Smart Tagging tables are fixed priors. Set-piece tags are split into quarters.
  A half is dead before its kickoff tag opens.
* **Robust to tracking noise:** speeds are measured over half a second, so position jitter
  cannot pass for movement, and a track jumping to another object is ignored.
* **Held-out results** (`scripts/benchmark.py`, mean spellF1; DFL under six tracking-problem
  profiles × three seeds: ball gaps of several kinds, a smoothed ball, and fragmented anonymous
  tracks with position noise):

  | model | Metrica | DFL dev | DFL reserved |
  |---|---|---|---|
  | first model (Metrica only, five tracking features) | 0.572 | 0.278 | 0.312 |
  | **submitted** | **0.658** | **0.455** | **0.495** |

  These show transfer between matches; they do not predict the competition score.

## Repository layout

```
run.py                        the one command (strict-external model)
possession/
  config.py                   every constant, each tagged with its source
  sources.py                  readers: tactical ATD, metadata, Smart Tagging
  features.py                 per-frame categorical features
  hmm.py                      3-state HMM tables and Viterbi decoding
  strict.py                   the submitted model: train on external games, apply frozen
  external.py                 readers + event-derived labels for the public sample games
  broadcast.py                broadcast-to-tactical alignment and ball fill
  touch.py                    touches from ball-velocity changes; the last-touch feature
  shape.py                    team-shape model (antisymmetric logistic regression)
  deadball.py                 still ball and restart kick (and "gap context", not adopted)
  dfl.py                      DFL open data reader; labels from the tracking
  degrade.py                  tracking-gap profiles for the external benchmark
  extensions.py               earlier candidate features, tested and not adopted (see report)
  pipeline.py                 data in, possession file out
  viewer.py                   self-contained HTML replay of tracking + labels
models/external_params.json   the frozen parameters (trained on external data)
scripts/train_external.py     regenerates models/external_params.json from the external data
scripts/download_dfl.py       downloads the DFL open data (CC BY 4.0)
scripts/benchmark.py          held-out benchmark on the external matches
scripts/test_extensions.py    leave-one-game-out test of candidate extra features (external data)
scripts/view_match.py         writes and opens the replay page (local only)
scripts/make_banner.py        draws docs/assets/pitch.svg, the simulated README banner
tests/                        synthetic-data tests, no match data needed: python -m pytest -q
evaluation/                   the organisers' validator and scorer, unchanged (MIT)
games.py                      the organisers' per-match constants, unchanged (MIT)
docs/TECHNICAL_REPORT.md      the technical document (rendered: docs/YamineLamal.pdf)
Dockerfile                    the pinned environment, as an alternative to pip
.github/workflows/tests.yml   runs the tests on every push
data/, external/, submission/ match data, external data, generated output: all gitignored
```

## Rules compliance, in short

* **Training and tuning on external data only.** Every learned or tuned number in the submitted
  model comes from Metrica's public sample games and the DFL open data. Nothing is fitted to the
  competition match.
* **The competition data is used as input only:** tracking, Smart Tagging and broadcast tracking.
  The videos are not used.
* **No ground truth.** No parameter was fitted, calibrated or chosen on the public ground truth,
  and the submitted model was never scored against it.
* **No hand labels or manual data** on the provided game.
* **No match data in this repository.** `data/`, `external/` and `submission/` are gitignored.

## Licence

`evaluation/` and `games.py` are the organisers' files, unchanged, under their MIT licence
(`LICENSE-organisers`). The rest of the code is the team's own. Neither the match data nor the
external data is covered by the organisers' licence. Metrica's sample data is acknowledged above.
The DFL open data is © Deutsche Fußball Liga, CC BY 4.0
(https://doi.org/10.6084/m9.figshare.28196177), described in Bassek, M., Rein, R., Weber, H.,
Memmert, D. (2025), *An integrated dataset of spatiotemporal and event data in elite soccer*,
Scientific Data 12, 195, doi:10.1038/s41597-025-04505-y. Neither dataset is redistributed here.
