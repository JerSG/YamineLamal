"""Every tunable number in the pipeline, in one place, with the reason it has the value it has.

None of these values was chosen by scoring against the public 5-minute ground truth. They come from
three sources only, and each constant says which:

  [football]  general knowledge about the game, true of any professional match
  [data]      a published fact about the provided files (DATA.md, FAQ.md, the ATD metadata)
  [model]     a modelling choice (smoothness, prior strength) that is not about this match

The emission tables are NOT here: they are learned by unsupervised EM on the whole match (see
hmm.py). What is here is only their starting point and how strongly it is held.
"""

FPS = 25

# -------------------------------------------------------------------------------------------------
# Transition model (fixed, not learned) -- per-frame probabilities derived from mean durations.
# -------------------------------------------------------------------------------------------------
# [football] A team spell with the ball alive lasts on average ~10-15 s in elite football once
# "controlled" possession is required (touches and deflections do not end it). 12 s is the middle.
MEAN_ALIVE_SPELL_S = 12.0
# [football] Roughly a third of possession spells end in a stoppage (ball out, foul, goal), the rest
# end in a turnover in open play.
P_SPELL_ENDS_IN_STOPPAGE = 0.33
# [football] A stoppage (throw-in, goal kick, foul, corner, injury, goal) lasts 10-30 s, longer for
# goals and injuries.
MEAN_DEAD_SPELL_S = 18.0

# [model] Emission tempering exponent -- the *maximum* per-feature weight. Each feature's actual
# weight is 1 / (mean run length of its codes), capped here; see hmm.PossessionHMM. Consecutive frames are far from independent (the nearest
# player to the ball does not change 25 times a second), so the HMM's per-frame evidence is raised
# to this power, which is equivalent to treating ~1 frame in 1/TEMPER as an independent observation.
# 0.2 = "one independent look every 5 frames" (0.2 s), the order of the tracker's own smoothing.
TEMPER = 0.2

# [model] Dirichlet prior strength for each learned emission table, in pseudo-frames. Small against
# the ~50k-frames-per-state of a full match, so the data decides; large enough that a rarely used
# feature value cannot collapse to 0 or 1.
PRIOR_STRENGTH = 2000.0
# [model] Bisection steps for the evidence scale (hmm.calibrate_scale): 10 halvings of a 256x
# range in log space pin the scale to within ~6%.
SCALE_SEARCH_STEPS = 10
# [model] Rounds of (calibrate scale, then EM), repeated until the scale moves by less than
# SCALE_TOLERANCE between rounds: EM sharpens the tables, which lowers the scale needed, so the
# two are iterated to a joint fixed point.
MAX_CALIBRATION_ROUNDS = 8
SCALE_TOLERANCE = 0.03
EM_ITERATIONS = 25
EM_TOLERANCE = 1e-6  # relative change of the tempered log-likelihood that ends EM early

# -------------------------------------------------------------------------------------------------
# Kickoffs and goals (Smart Tagging structure)
# -------------------------------------------------------------------------------------------------
# [data] Kickoff tags are fixed 5 s windows. The two published kickoff frames (DATA.md: 11185 and
# 113742) sit 88 and 82 frames after their tags' starts, so the ball is struck ~85 frames in.
# Everything in a half before that point is a dead ball by definition.
KICKOFF_STRIKE_OFFSET = 85

# [football] Goal feature: seconds remaining until the restart kickoff, binned. A goal tag runs from
# the attack that produced the goal to the kickoff after it; the celebration (and any VAR check)
# typically takes the last 40-120 s of it. The bins let EM learn where the dead ball starts.
GOAL_BINS_S = (30.0, 60.0, 120.0)

# -------------------------------------------------------------------------------------------------
# Set pieces (Smart Tagging)
# -------------------------------------------------------------------------------------------------
# [data] Non-kickoff set-piece tags are fixed 35 s windows; the restart sits somewhere inside. Bins
# of time since the window opened, so EM can learn where "dead" turns into "alive" on average.
SETPIECE_BINS_S = (3.0, 8.0, 15.0)

# -------------------------------------------------------------------------------------------------
# Ball feature
# -------------------------------------------------------------------------------------------------
# [model] Length of the ball-missing stretch a frame sits in. The ball is missing in long blocks
# (median ~11 s), during play and stoppages alike, so length is evidence, not a verdict.
GAP_BINS_S = (1.0, 4.0, 10.0)
# [football] A player is "on the ball" within ~2 m of it (tracker noise included).
CONTROL_RADIUS_M = 2.0

# -------------------------------------------------------------------------------------------------
# Kinematic features (full tracking only)
# -------------------------------------------------------------------------------------------------
# [data] The ATD metadata labels player speed "kph", but the values are m/s: median 1.3, 95th
# percentile 4.6, 99.9th 8.1 -- in km/h that would be a match played at walking pace. DATA.md says
# m/s, and the numbers agree with it.
SPEED_UNIT_TO_MS = 1.0
# [football] Mean speed of the tracked players: a stoppage is walked (< ~1 m/s), open play is
# jogged and run (~1.5-2.5 m/s mean over 22 players).
PLAYER_SPEED_BINS_MS = (0.9, 1.3, 1.8, 2.5)
SPEED_SMOOTH_FRAMES = 12  # +/- 0.5 s centred mean
MIN_TRACKED_PLAYERS = 8   # below this the mean speed is not trusted
# [football] Ball speed from consecutive positions: placed/still, rolling/dribbled, passed, struck.
BALL_SPEED_BINS_MS = (0.7, 4.0, 12.0)
BALL_SPEED_SMOOTH_FRAMES = 3
# [data/football] Ball position against the lines, in metres. The ball is out once wholly over the
# line; 0.5 m of margin absorbs tracking error.
OUT_MARGIN_M = 0.5
EDGE_MARGIN_M = 1.5
# [football] The team in possession spreads out, the defending team compacts. log(area ratio) bins.
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
