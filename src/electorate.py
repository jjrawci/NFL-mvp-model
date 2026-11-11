"""
electorate.py
-------------
Turns ranker scores into win probabilities by simulating the AP MVP vote.

Rather than a softmax over scores, each simulation draws a 50-voter electorate:

  utility(voter, player) = z_player + shared_shock(player) + voter_noise

  z_player      the ranker score, standardised within the contender pool
  shared_shock  N(0, sigma), common to all voters in one simulation. This is
                the "narrative" that moves the whole electorate together
                (a comeback story, a late-season surge). Larger sigma means
                more upsets and wider odds on the favourite.
  voter_noise   N(0, tau), independent per voter. This is individual
                disagreement. It splits points between close candidates.

Each voter ranks their top 5 and awards 10-5-3-2-1 points (the AP scheme).
Most points wins; exact ties are broken at random. A player's win probability
is the share of simulations they win.

sigma and tau are fitted by `fit_params` against past seasons.
"""

import numpy as np

N_VOTERS = 50
POINTS = np.array([10, 5, 3, 2, 1], dtype=float)
POOL_SIZE = 15
PROB_FLOOR = 1e-3


def pool_zscores(scores: np.ndarray) -> np.ndarray:
    """Standardise scores within the contender pool (same convention as predict)."""
    s = np.asarray(scores, dtype=float)
    return (s - s.mean()) / (s.std() + 1e-9)


def simulate_win_probs(z: np.ndarray, sigma: float, tau: float,
                       n_sims: int = 4000, seed: int = 0) -> np.ndarray:
    """
    Win probability for each pool member given standardised scores `z`.
    Uses a fixed seed so results are reproducible and comparable across
    parameter settings (common random numbers).
    """
    z = np.asarray(z, dtype=float)
    n = len(z)
    rng = np.random.default_rng(seed)
    shock = rng.normal(0.0, 1.0, size=(n_sims, 1, n)) * sigma
    noise = rng.normal(0.0, 1.0, size=(n_sims, N_VOTERS, n)) * tau
    util = z[None, None, :] + shock + noise                  # (sims, voters, players)

    k = min(len(POINTS), n)
    top = np.argsort(-util, axis=2)[:, :, :k]                # each voter's top picks
    totals = np.zeros((n_sims, n))
    for j in range(k):
        np.add.at(totals, (np.arange(n_sims)[:, None], top[:, :, j]), POINTS[j])

    # tiny random jitter breaks exact ties uniformly
    totals = totals + rng.random(totals.shape) * 1e-6
    winners = totals.argmax(axis=1)
    return np.bincount(winners, minlength=n) / n_sims
