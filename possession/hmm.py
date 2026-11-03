"""A three-state hidden Markov model of possession: H (home has it), A (away has it), D (dead).

    state:        H ---- A          fixed transition model from football priors (config.py):
                   \\    /           a spell lasts ~12 s, a third of spells end in a stoppage,
                    \\  /            a stoppage lasts ~18 s, a restart goes to either team.
                     D

    emissions:    naive-Bayes product of categorical features (features.py), each raised to the
                  tempering exponent because consecutive frames are not independent looks.

Two tyings keep the model honest with no labels to learn from:
  * H and A share one table, mirrored home<->away. A feature cannot mean "Barcelona" for one state
    and something unrelated for the other -- only "the team in possession" vs "the other team".
  * D is mirror-symmetric: a dead ball favours neither team.

Identifiability. With no labels, EM is free to make "D" mean whatever cluster explains the data
best -- and the first version did exactly that: D became "no Smart Tagging phase is running" and
"camera on the middle third". So the dead-ball table is learned only for features that physically
measure a stoppage (ball-missing stretches; in full mode player speed, ball speed, ball against the
lines). Every other feature keeps a fixed, mild prior table for D, and slow, long-correlated context
features (camera zone, goal timing) keep fixed tables for both states.

Learning is tempered Baum-Welch on the emission tables only, over the whole match, with a Dirichlet
prior centred on each feature's starting table. No ground truth is read anywhere in this module.
Decoding is Viterbi, because the metric scores whole possessions and Viterbi returns the single
most likely *segmentation*, not a frame-by-frame majority.
"""

import math

import numpy as np

from . import config as C

H, A, D = 0, 1, 2
N_STATES = 3
STATE_LABELS = np.array(["H", "A", "D"])


def mean_run_length(codes, blocks=None):
    """Mean length, in frames, of the runs of identical codes (within blocks if given)."""
    spans = blocks or [(0, len(codes))]
    runs = total = 0
    for s, e in spans:
        c = codes[s:e]
        if len(c):
            runs += 1 + int(np.count_nonzero(c[1:] != c[:-1]))
            total += len(c)
    return total / float(max(runs, 1))


def transition_matrix(fps=None):
    fps = fps or C.FPS
    leave_alive = 1.0 / (C.MEAN_ALIVE_SPELL_S * fps)
    leave_dead = 1.0 / (C.MEAN_DEAD_SPELL_S * fps)
    to_dead = leave_alive * C.P_SPELL_ENDS_IN_STOPPAGE
    to_other = leave_alive * (1.0 - C.P_SPELL_ENDS_IN_STOPPAGE)
    return np.array([
        [1.0 - leave_alive, to_other, to_dead],
        [to_other, 1.0 - leave_alive, to_dead],
        [leave_dead / 2.0, leave_dead / 2.0, 1.0 - leave_dead],
    ])


class PossessionHMM(object):
    def __init__(self, features, temper=None, prior_strength=None, blocks=None):
        self.features = features
        temper = C.TEMPER if temper is None else temper
        prior_strength = C.PRIOR_STRENGTH if prior_strength is None else prior_strength
        self.temper = temper
        # Per-feature tempering: a feature counts once per *independent look*, and the length of a
        # look is measured from the data as the mean run length of that feature's codes on graded
        # frames (never below 1 / temper). A Smart Tagging phase that holds for 20 s contributes
        # roughly one observation per 20 s, not 500. Unsupervised: it reads only the features.
        self.base_weights = np.array([min(temper, 1.0 / mean_run_length(f.codes, blocks))
                                      for f in features])
        # One global multiplier on top, set by calibrate_scale(). The per-feature weights fix
        # how features trade off against each other; the scale fixes evidence vs. transitions.
        self.scale = 1.0
        self.prior_strength = prior_strength
        self.trans = transition_matrix()
        self.log_trans = np.log(self.trans)
        self.tables = [self._tables_from(f.init_team, f.init_dead, f) for f in features]

    @property
    def weights(self):
        return self.base_weights * self.scale

    # -- parameters ---------------------------------------------------------------------------
    @staticmethod
    def _tables_from(team, dead, feature):
        """(3, V) emission table: H = team, A = mirrored team, D = symmetrised dead."""
        dead = feature.renormalise(0.5 * (dead + dead[feature.swap]))
        team = feature.renormalise(team)
        return np.vstack([team, team[feature.swap], dead])

    def log_emissions(self, allowed=None):
        T = len(self.features[0].codes)
        L = np.zeros((T, N_STATES))
        for f, table, w in zip(self.features, self.tables, self.weights):
            contribution = np.log(table[:, f.codes].T)
            contribution[f.neutral[f.codes]] = 0.0  # no measurement, no evidence
            L += w * contribution
        if allowed is not None:
            L = np.where(allowed, L, -np.inf)
        return L

    # -- inference ----------------------------------------------------------------------------
    def forward_backward(self, L, blocks):
        """Posterior state probabilities (T, 3) over the given [start, end) blocks; 0 elsewhere.

        Scaled recursion in plain Python floats: with three states, scalar arithmetic beats numpy's
        per-call overhead by an order of magnitude.
        """
        gamma = np.zeros_like(L)
        total_loglik = 0.0
        tr = self.trans.tolist()
        for start, end in blocks:
            Lb = L[start:end]
            peak = Lb.max(axis=1, keepdims=True)
            E = np.exp(Lb - peak).tolist()
            n = end - start
            alpha = [None] * n
            scale = [0.0] * n
            e = E[0]
            a = [e[0] / 3.0, e[1] / 3.0, e[2] / 3.0]
            s = a[0] + a[1] + a[2]
            alpha[0] = [a[0] / s, a[1] / s, a[2] / s]
            scale[0] = s
            (t00, t01, t02), (t10, t11, t12), (t20, t21, t22) = tr
            for t in range(1, n):
                p0, p1, p2 = alpha[t - 1]
                e = E[t]
                a0 = (p0 * t00 + p1 * t10 + p2 * t20) * e[0]
                a1 = (p0 * t01 + p1 * t11 + p2 * t21) * e[1]
                a2 = (p0 * t02 + p1 * t12 + p2 * t22) * e[2]
                s = a0 + a1 + a2
                alpha[t] = [a0 / s, a1 / s, a2 / s]
                scale[t] = s
            g = np.empty((n, 3))
            b0 = b1 = b2 = 1.0
            g[n - 1] = alpha[n - 1]
            for t in range(n - 2, -1, -1):
                e = E[t + 1]
                w0, w1, w2 = b0 * e[0], b1 * e[1], b2 * e[2]
                s = scale[t + 1]
                b0 = (t00 * w0 + t01 * w1 + t02 * w2) / s
                b1 = (t10 * w0 + t11 * w1 + t12 * w2) / s
                b2 = (t20 * w0 + t21 * w1 + t22 * w2) / s
                a = alpha[t]
                q0, q1, q2 = a[0] * b0, a[1] * b1, a[2] * b2
                z = q0 + q1 + q2
                g[t, 0], g[t, 1], g[t, 2] = q0 / z, q1 / z, q2 / z
            gamma[start:end] = g
            total_loglik += float(np.sum(np.log(scale)) + peak.sum())
        return gamma, total_loglik

    def viterbi(self, L, blocks):
        """Most likely state path over each block; -1 outside the blocks."""
        path = np.full(len(L), -1, dtype=np.int8)
        lt = self.log_trans.tolist()
        for start, end in blocks:
            Lb = L[start:end].tolist()
            n = end - start
            back = np.zeros((n, 3), dtype=np.int8)
            log_third = math.log(1.0 / 3.0)
            d = [log_third + x for x in Lb[0]]
            for t in range(1, n):
                e = Lb[t]
                nd = [0.0, 0.0, 0.0]
                bk = [0, 0, 0]
                for j in range(3):
                    c0 = d[0] + lt[0][j]
                    c1 = d[1] + lt[1][j]
                    c2 = d[2] + lt[2][j]
                    if c0 >= c1 and c0 >= c2:
                        best, arg = c0, 0
                    elif c1 >= c2:
                        best, arg = c1, 1
                    else:
                        best, arg = c2, 2
                    nd[j] = best + e[j]
                    bk[j] = arg
                back[t] = bk
                d = nd
            state = int(np.argmax(d))
            out = np.empty(n, dtype=np.int8)
            for t in range(n - 1, -1, -1):
                out[t] = state
                state = back[t, state]
            path[start:end] = out
        return path

    # -- learning -----------------------------------------------------------------------------
    def m_step(self, gamma, blocks):
        mask = np.zeros(len(gamma), dtype=bool)
        for s, e in blocks:
            mask[s:e] = True
        new = []
        for f in self.features:
            V = len(f.values)
            c = [np.bincount(f.codes[mask], weights=gamma[mask, k], minlength=V) for k in range(3)]
            prior_team = np.where(f.neutral, 0.0, f.init_team)
            prior_dead = np.where(f.neutral, 0.0, f.init_dead)
            team = c[H] + c[A][f.swap] + self.prior_strength * prior_team \
                if f.learn_team else f.init_team
            dead = c[D] + self.prior_strength * prior_dead if f.learn_dead else f.init_dead
            new.append(self._tables_from(team, dead, f))
        self.tables = new

    def fit(self, blocks, allowed=None, iterations=None, tol=None, verbose=True):
        iterations = C.EM_ITERATIONS if iterations is None else iterations
        tol = C.EM_TOLERANCE if tol is None else tol
        previous = None
        for it in range(iterations):
            gamma, loglik = self.forward_backward(self.log_emissions(allowed), blocks)
            self.m_step(gamma, blocks)
            if verbose:
                occupancy = gamma[np.concatenate([np.arange(s, e) for s, e in blocks])].mean(0)
                print("  EM {:>2}  tempered loglik {:>14.1f}   occupancy H {:.3f} A {:.3f} "
                      "D {:.3f}".format(it + 1, loglik, *occupancy))
            if previous is not None and abs(loglik - previous) <= tol * abs(previous):
                break
            previous = loglik
        return self

    def decode(self, blocks, allowed=None):
        return self.viterbi(self.log_emissions(allowed), blocks)

    def mean_alive_spell_s(self, blocks, allowed=None):
        path = self.decode(blocks, allowed)
        lengths = []
        for s, e in blocks:
            p = path[s:e]
            change = np.flatnonzero(p[1:] != p[:-1]) + 1
            for a, b in zip(np.concatenate(([0], change)), np.concatenate((change, [len(p)]))):
                if p[a] != D:
                    lengths.append(b - a)
        return np.mean(lengths) / float(C.FPS) if lengths else float("inf")

    def calibrate_scale(self, blocks, allowed=None, target_s=None, lo=0.25, hi=64.0, steps=None,
                        verbose=True):
        """Self-consistency: choose the evidence scale at which the decoded possessions have the
        mean length the transition model itself assumes. More evidence -> more, shorter spells,
        so the decoded mean falls monotonically (up to decoding noise) as the scale grows and a
        geometric bisection finds the crossing. Reads no labels."""
        target_s = C.MEAN_ALIVE_SPELL_S if target_s is None else target_s
        steps = C.SCALE_SEARCH_STEPS if steps is None else steps
        for _ in range(steps):
            mid = math.sqrt(lo * hi)
            self.scale = mid
            got = self.mean_alive_spell_s(blocks, allowed)
            if verbose:
                print("  scale {:>7.3f}  decoded mean spell {:>5.1f} s  (target {:.1f})".format(
                    mid, got, target_s))
            if got > target_s:
                lo = mid
            else:
                hi = mid
        self.scale = math.sqrt(lo * hi)
        return self.scale

    def describe(self):
        lines = []
        for f, table, w in zip(self.features, self.tables, self.weights):
            lines.append("  {:<9} {:<16} {:>8} {:>8}   weight {:.4f} (1 look per {:.0f} frames)"
                         .format(f.name, "value", "P(.|H)", "P(.|D)", w, 1.0 / w))
            for i, v in enumerate(f.values):
                if f.neutral[i]:
                    lines.append("  {:<9} {:<16} {:>8} {:>8}".format("", v, "neutral", "neutral"))
                    continue
                lines.append("  {:<9} {:<16} {:>8.4f} {:>8.4f}".format("", v, table[H, i],
                                                                      table[D, i]))
        return "\n".join(lines)
