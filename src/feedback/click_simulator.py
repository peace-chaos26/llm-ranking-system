# src/feedback/click_simulator.py
"""
Simulates user clicks on ranked results.

Click model:
  P(click | rank, relevant) = P(examined | rank) * P(click | examined, relevant)

Where:
  P(examined | rank)          = position bias (propensity)
  P(click | examined, rel=1)  = click_prob_relevant   (e.g. 0.8)
  P(click | examined, rel=0)  = click_prob_irrelevant (e.g. 0.1)
"""

import numpy as np
from dataclasses import dataclass

from src.feedback.position_bias import PositionBiasModel


@dataclass
class ClickEvent:
    query_id: str
    passage_id: str
    rank: int                   # position shown (1-indexed)
    was_examined: bool          # did user scroll to this position?
    was_clicked: bool           # did user click?
    is_relevant: bool           # ground truth (unknown to the system)
    ips_weight: float           # inverse propensity weight for debiasing


class ClickSimulator:

    def __init__(
        self,
        position_bias: PositionBiasModel,
        click_prob_relevant: float = 0.8,
        click_prob_irrelevant: float = 0.1,
        seed: int = 42,
    ):
        self.bias_model = position_bias
        self.click_prob_relevant = click_prob_relevant
        self.click_prob_irrelevant = click_prob_irrelevant
        np.random.seed(seed)

    def simulate_session(
        self,
        query_id: str,
        ranked_results: list[tuple[str, float]],
        relevant_passages: dict[str, int],
        n_impressions: int = 10,
    ) -> list[ClickEvent]:
        """
        Simulate one user session on a ranked result list.

        Args:
            query_id: query identifier
            ranked_results: [(passage_id, score), ...] sorted by rank
            relevant_passages: {passage_id: relevance} ground truth
            n_impressions: how many results are shown (above the fold)

        Returns: list of ClickEvents, one per shown result
        """
        events = []
        results_shown = ranked_results[:n_impressions]

        for rank, (pid, score) in enumerate(results_shown, start=1):
            is_relevant = relevant_passages.get(pid, 0) > 0

            # Was this position examined?
            prop = self.bias_model.propensity(rank)
            examined = np.random.random() < prop

            # Given examination, was it clicked?
            if examined:
                click_prob = (
                    self.click_prob_relevant
                    if is_relevant
                    else self.click_prob_irrelevant
                )
                clicked = np.random.random() < click_prob
            else:
                clicked = False

            events.append(ClickEvent(
                query_id=query_id,
                passage_id=pid,
                rank=rank,
                was_examined=examined,
                was_clicked=clicked,
                is_relevant=is_relevant,
                ips_weight=self.bias_model.ips_weight(rank),
            ))

        return events

    def simulate_batch(
        self,
        queries: dict[str, str],
        ranked_results: dict[str, list[tuple[str, float]]],
        qrels: dict[str, dict[str, int]],
        n_sessions_per_query: int = 50,
        n_impressions: int = 10,
    ) -> list[ClickEvent]:
        """
        Simulate multiple user sessions per query.
        More sessions = better signal, less noise.

        n_sessions_per_query=50 means each query is "shown" to 50 users.
        This gives stable click statistics for retraining.
        """
        all_events = []

        for qid in queries:
            if qid not in ranked_results or qid not in qrels:
                continue

            for _ in range(n_sessions_per_query):
                events = self.simulate_session(
                    query_id=qid,
                    ranked_results=ranked_results[qid],
                    relevant_passages=qrels[qid],
                    n_impressions=n_impressions,
                )
                all_events.extend(events)

        return all_events