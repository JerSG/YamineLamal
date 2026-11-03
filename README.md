# Ball possession from automatic tracking: a self-calibrating HMM (team YamineLamal)

Team **YamineLamal** submission for **More than a Hack 2027, Phase 1: Ball Possession**
(Metrica Sports · FC Barcelona · Mobile World Capital Barcelona).

For every frame of FC Barcelona vs Athletic Club, the pipeline outputs `H`, `A` or `D`. It uses
a three-state hidden Markov model whose emission tables are learned by **unsupervised EM on the
whole match**, run on **two-camera fused tracking**: the broadcast recording is aligned to the
tactical timeline automatically, shot by shot, and fills in the ball where the tactical tracker
lost it. No ground truth is read at any point, and no constant is tied to a frame of the
game. The approach is described in [`docs/TECHNICAL_REPORT.md`](docs/TECHNICAL_REPORT.md).

```
tactical tracking ──────────────┐
broadcast tracking ─> align ─> fill ball ─┤
Smart Tagging ──────────────────┼─> 9 features ─> tempered EM (no labels) ─> Viterbi ─> H/A/D
tactical homography ────────────┘                 + self-consistent evidence scale
```

## Reproduce the submission

**Python 3.12 or newer** (verified on 3.13): the pinned numpy 2.5.3 needs 3.12+. numpy is the only
runtime dependency of the pipeline; pandas is for the organisers' scorer.

```bash
pip install -r requirements.txt
```

Unpack the match data into `data/` with the names it ships with (see [`data/README.md`](data/README.md)).
The submission was generated from these seven files. No video is needed:

```
data/tactical/BAR-ATH_tactical_atd.xml
data/tactical/BAR-ATH_tactical_atd.txt
data/tactical/BAR-ATH_tactical_smart_tagging.xml
data/tactical/BAR-ATH_tactical_homography.csv
data/broadcast/BAR-ATH_broadcast_atd.xml
data/broadcast/BAR-ATH_broadcast_atd.txt
data/broadcast/BAR-ATH_broadcast_homography.csv
```

```bash
python run.py --team YamineLamal
```

This takes about 2 minutes on a 4-core laptop (the broadcast alignment runs on all CPU cores) and writes `submission/YamineLamal.csv` and
`submission/YamineLamal_report.json`. The run is deterministic: the same data and the same command
give a byte-identical file.

```bash
python evaluation/validate_submission.py submission/YamineLamal.csv
```

Optionally, score it on the public 5-minute window. That window is a pipeline check, not a
leaderboard:

```bash
cp submission/YamineLamal.csv evaluation/results/
python evaluation/run_evaluation.py
```

### Options

| flag | effect |
|---|---|
| `--mode auto` (default) | `full` if `BAR-ATH_tactical_atd.txt` is present, else `basic` |
| `--mode full` | raw tracking: nearest-player distances, player speed, ball speed, ball against the lines, team spread |
| `--mode basic` | no raw tracking needed. Uses the organisers' published nearest-player signal (`evaluation/results/baseline_raw.csv`, computed from the ATD by their `starter/baseline.py`) |
| `--no-broadcast` | skip broadcast fusion (about 1 minute). This reproduces our earlier tactical-only file |
| `--no-camera` | ignore the tactical homography |

Broadcast fusion switches on automatically when the three broadcast files are present. The
submitted file was generated with the plain command above and all seven files: `mode: full +
broadcast fusion`.

## Repository layout

```
run.py                      the one command
possession/
  config.py                 every tunable number, each tagged [football] / [data] / [model]
  sources.py                readers: raw ATD, ATD metadata, Smart Tagging, published nearest signal
  features.py               per-frame categorical features
  hmm.py                    tied 3-state HMM, tempered EM, Viterbi, evidence-scale calibration
  broadcast.py              broadcast-to-tactical alignment (shots, offset search, checks) and ball fill
  pipeline.py               data in, possession file out
scripts/sensitivity.py      unsupervised robustness check: how much the output moves per prior
tests/                      synthetic-data tests (no match data needed): python -m pytest -q
evaluation/                 the organisers' scorer, unchanged (MIT), with the public 5-min GT
games.py                    the organisers' per-match constants, unchanged (MIT)
docs/TECHNICAL_REPORT.md    the technical document (rendered: docs/YamineLamal.pdf)
data/                       the match data goes here, gitignored
submission/                 generated output, gitignored
```

## Rules compliance, in short

- **No hand labels and no human-produced data about this game.** All labels come from the
  algorithm. Emission tables are learned by unsupervised EM.
- **The public 5-minute ground truth was not used for training or for choosing parameters.**
  It was used to confirm the file format and frame alignment, and as an occasional sanity check
  while debugging. The report lists exactly what was checked and why the final parameters do
  not depend on it.
- **No match data in this repository.** `data/` and `submission/` are gitignored.

## Licence

Our code is released under the MIT licence (`LICENSE`). `evaluation/` and `games.py` are the organisers'
files, unchanged, under their MIT licence (see `LICENSE-organisers`). The match data is not covered by
either licence.
