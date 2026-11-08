"""A three-state hidden Markov model of possession: H (home has it), A (away has it), D (dead).

    state:        H ---- A          transition probabilities learned on other matches
                   \\    /           (models/external_params.json, see strict.py)
                    \\  /
                     D

    emissions:    naive-Bayes product of categorical features (features.py), one table per
                  feature, each weighted so that a feature counts once per independent look.

Two tyings keep the model about "the team in possession" rather than about a particular club:
  * H and A share one table, mirrored home<->away. A feature cannot mean "Barcelona" for one state
    and something unrelated for the other -- only "the team in possession" vs "the other team".
  * D is mirror-symmetric: a dead ball favours neither team.

Decoding is Viterbi, because the metric scores whole possessions and Viterbi returns the single
most likely *segmentation*, not a frame-by-frame majority. This module holds no parameters of its
own: the tables, transitions and evidence scale are supplied, frozen (strict.FrozenHMM).
"""

import math

import numpy as np

H, A, D = 0, 1, 2
N_STATES = 3
STATE_LABELS = np.array(["H", "A", "D"])


class PossessionHMM(object):
    """Shared machinery: tied tables, emissions and Viterbi. strict.FrozenHMM supplies the
    parameters (tables, transitions, per-frame weights, scale)."""

    @staticmethod
    def _tables_from(team, dead, feature):
        """(3, V) emission table: H = team, A = mirrored team, D = symmetrised dead."""
        dead = feature.renormalise(0.5 * (dead + dead[feature.swap]))
        team = feature.renormalise(team)
        return np.vstack([team, team[feature.swap], dead])

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

    def decode(self, blocks, allowed=None):
        return self.viterbi(self.log_emissions(allowed), blocks)
