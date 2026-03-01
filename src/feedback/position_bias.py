# src/feedback/position_bias.py
"""
Position bias model for click simulation.

In real systems, position bias is estimated via:
- Randomization experiments (swap positions, measure click delta)
- Eye-tracking studies
- EM algorithm on click logs (Joachims et al.)

We simulate it using a power-law decay — well-established in
IR literature as a good approximation of real user behavior.

Propensity model:
  P(examined | rank) = (1 / rank) ^ eta

  eta = 0: no position bias (users examine everything equally)
  eta = 1: strong position bias (standard assumption)
  eta > 1: very strong bias (mobile search behavior)

Inverse Propensity Score (IPS):
  weight(click at rank r) = 1 / P(examined | rank=r)

Why IPS works:
If a document at rank 5 has propensity 0.2 (examined 20% of the time),
a click on it carries weight 1/0.2 = 5x more signal than a click at
rank 1 (propensity ~1.0, weight ~1.0).
This recovers an unbiased estimate of true relevance from biased clicks.
"""

import numpy as np


class PositionBiasModel:

    def __init__(self, eta: float = 1.0):
        """
        eta: position bias strength
             1.0 = standard web search bias
             0.5 = mild bias (navigational queries)
             1.5 = strong bias (mobile/voice)
        """
        self.eta = eta

    def propensity(self, rank: int) -> float:
        """
        P(user examines result at rank).
        Rank is 1-indexed.
        """
        return 1.0 / (rank ** self.eta)

    def propensities(self, n: int) -> np.ndarray:
        """Return propensity for each rank 1..n."""
        return np.array([self.propensity(r) for r in range(1, n + 1)])

    def ips_weight(self, rank: int) -> float:
        """
        Inverse propensity weight for a click at this rank.
        Higher rank (lower position) = higher weight.
        Clipped to avoid extreme weights for very low positions.
        """
        return min(1.0 / self.propensity(rank), 10.0)  # clip at 10x

    def simulate_examination(self, n_results: int) -> np.ndarray:
        """
        Simulate whether each position was examined by a user.
        Returns binary array of length n_results.
        """
        props = self.propensities(n_results)
        return (np.random.random(n_results) < props).astype(int)