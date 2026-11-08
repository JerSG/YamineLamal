"""Every constant in the pipeline, in one place, with where its value comes from.

No value here was chosen by scoring against any competition data. Each constant says its source:

  [football]  general knowledge about the game, true of any professional match
  [data]      a published fact about the provided files' format (DATA.md, the ATD metadata)
  [model]     a modelling choice (smoothness, prior strength) that is not about this match

What the model LEARNS is not here: the feature bin edges, emission tables, transition
probabilities and evidence scale are learned on the external games and frozen in
models/external_params.json (strict.py). Values below marked as bin edges are only defaults that
those learned edges replace.
"""

FPS = 25

# [model] Dirichlet prior strength for each emission table fitted on the external games, in
# pseudo-frames: small against the external games' ~100k frames per state, large enough that a
# rarely seen feature value cannot collapse to probability 0 or 1.
PRIOR_STRENGTH = 2000.0

# -------------------------------------------------------------------------------------------------
# Goals (Smart Tagging)
# -------------------------------------------------------------------------------------------------
# [football] Goal feature: seconds remaining until the kickoff tag after the goal, binned. A goal
# tag runs from the attack that produced the goal to the kickoff after it; the celebration (and
# any VAR check) typically takes the last 40-120 s of it.
GOAL_BINS_S = (30.0, 60.0, 120.0)

# -------------------------------------------------------------------------------------------------
# Set pieces (Smart Tagging)
# -------------------------------------------------------------------------------------------------
# Default bins of time since a set-piece tag opened. Not used by the submitted model, which splits
# each tag into quarters of its own length instead (strict.SETPIECE_FRACTIONS).
SETPIECE_BINS_S = (3.0, 8.0, 15.0)

# -------------------------------------------------------------------------------------------------
# Ball feature
# -------------------------------------------------------------------------------------------------
# Bin edges for the length of the ball-missing stretch a frame sits in. DEFAULT ONLY: replaced by
# quartiles of the external games (models/external_params.json, "bins").
GAP_BINS_S = (1.0, 4.0, 10.0)
# [football] A player is "on the ball" within ~2 m of it (tracker noise included).
CONTROL_RADIUS_M = 2.0

# -------------------------------------------------------------------------------------------------
# Kinematic features (from the tracking)
# -------------------------------------------------------------------------------------------------
# [data] Unit of the tracking's speed channel (DATA.md: m/s). The submitted model does not read
# that channel: it recomputes speeds from positions (strict.speed_from_positions), identically
# for the external games and the match.
SPEED_UNIT_TO_MS = 1.0
# Bin edges for the mean speed of the tracked players (a stoppage is walked). DEFAULT ONLY:
# replaced by fifths of the external games.
PLAYER_SPEED_BINS_MS = (0.9, 1.3, 1.8, 2.5)
SPEED_SMOOTH_FRAMES = 12  # +/- 0.5 s centred mean
MIN_TRACKED_PLAYERS = 8   # below this the mean speed is not trusted
# Bin edges for ball speed from consecutive positions. DEFAULT ONLY: replaced by quartiles of the
# external games.
BALL_SPEED_BINS_MS = (0.7, 4.0, 12.0)
BALL_SPEED_SMOOTH_FRAMES = 3
# [data/football] Ball position against the lines, in metres. The ball is out once wholly over the
# line; 0.5 m of margin absorbs tracking error.
OUT_MARGIN_M = 0.5
EDGE_MARGIN_M = 1.5
# |log(area ratio)| below which the two teams count as equally spread (the team in possession
# spreads out, the defending team compacts). DEFAULT ONLY: replaced by the external games' third.
SPREAD_LOG_RATIO = 0.25

# -------------------------------------------------------------------------------------------------
# Broadcast alignment (possession/broadcast.py) -- geometry only, no labels
# -------------------------------------------------------------------------------------------------
# [football] A broadcast camera pans at most a few metres of pitch per frame (25 fps); a larger
# jump of its aim point is a cut to another shot.
BC_CUT_JUMP_M = 4.0
BC_MIN_SHOT_FRAMES = 50       # [model] shots shorter than 2 s are not worth aligning
BC_MIN_PLAYERS = 4            # [model] frames with fewer tracked players carry no shape to match
BC_SAMPLES_PER_SHOT = 6       # [model] sample frames per shot for the offset search
BC_MATCH_CAP_M = 5.0          # [model] cap on one player's mismatch (partial views, tracker errors)
BC_RUNNER_UP_EXCLUSION = 50   # [model] offsets within 2 s of the best do not count as rivals
BC_COARSE_STEP = 5            # [model] coarse offset grid (0.2 s), then refined frame by frame
BC_REFINE_CANDIDATES = 8      # [model] coarse minima refined at full resolution
BC_WORKERS = 0                # processes for the offset search; 0 = one per CPU core (max 8).
                              # Changes only the run time, never the result.
# [model] Acceptance: players fit within ~1.5 m on average, the best offset beats any other offset
# by 1 m, and where both trackers see the ball the two positions agree (median) within 2 m.
BC_MAX_COST_M = 1.5
BC_MIN_MARGIN_M = 1.0
BC_MIN_BALL_CHECK_FRAMES = 10
BC_MAX_BALL_ERR_M = 2.0

# -------------------------------------------------------------------------------------------------
# Strict-external model (possession/strict.py): parameters learned on other matches only
# -------------------------------------------------------------------------------------------------
STRICT_EXTERNAL_PITCH = (105.0, 68.0)   # [data] Metrica sample games' pitch (their README)
STRICT_SPEED_SMOOTH_FRAMES = 5          # [model] +/- 0.2 s smoothing of position-derived speed
# [model] evidence scales tried on the external games; the best mean spellF1 there is kept
STRICT_SCALE_GRID = (2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 12.0, 14.0, 16.0)
# [model] Most observations a feature may contribute: one per second (25 frames), however much
# it flickers. Protects the frozen model from noisier tracking than it was trained on.
STRICT_MIN_LOOK_FRAMES = 25
