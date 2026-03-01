# src/feedback/feedback_processor.py
"""
Processes click events into training signal for ranker retraining.

Two approaches implemented:
1. Naive: treat clicks as relevance labels directly
   Problem: encodes position bias — model learns to rank position 1 high
   
2. IPS-debiased: weight clicks by inverse propensity
   Recovers unbiased relevance signal from biased clicks
   Standard approach in production LTR systems
"""

import numpy as np
from collections import defaultdict
from loguru import logger

from src.feedback.click_simulator import ClickEvent
from src.ranking.features import compute_features


class FeedbackProcessor:

    def __init__(self, use_ips: bool = True):
        """
        use_ips: if True, apply IPS debiasing
                 if False, use naive click = relevance
        """
        self.use_ips = use_ips

    def clicks_to_labels(
        self,
        events: list[ClickEvent],
    ) -> dict[str, dict[str, float]]:
        """
        Aggregate click events into pseudo-relevance labels.

        For each (query, passage) pair:
        - Naive: label = click_rate (fraction of sessions where clicked)
        - IPS:   label = sum(click * ips_weight) / sum(ips_weight)

        IPS formula:
          label_ips(q, d) = Σ(click_i * w_i) / Σ(w_i)
          where w_i = 1/propensity(rank_i)

        This is importance-weighted averaging — standard in causal ML.
        """
        # Accumulate per (query, passage)
        click_weighted = defaultdict(float)
        weight_sum = defaultdict(float)
        click_count = defaultdict(int)
        impression_count = defaultdict(int)

        for event in events:
            key = (event.query_id, event.passage_id)
            impression_count[key] += 1

            if self.use_ips:
                weight = event.ips_weight
            else:
                weight = 1.0

            weight_sum[key] += weight
            if event.was_clicked:
                click_weighted[key] += weight
                click_count[key] += 1

        # Convert to pseudo-labels per query
        labels: dict[str, dict[str, float]] = defaultdict(dict)
        for (qid, pid), total_weight in weight_sum.items():
            if total_weight > 0:
                raw = click_weighted[(qid, pid)] / total_weight
            else:
                raw = 0.0
            # LambdaMART requires integer labels.
            # Bucket continuous IPS scores into 0/1/2 relevance grades:
            # 0 = not clicked, 1 = weakly clicked, 2 = strongly clicked
            if raw >= 0.5:
                labels[qid][pid] = 2
            elif raw >= 0.1:
                labels[qid][pid] = 1
            else:
                labels[qid][pid] = 0

        logger.info(
            f"Processed {len(events)} click events → "
            f"{sum(len(v) for v in labels.values())} (query, passage) labels"
        )
        return dict(labels)

    def build_training_features(
        self,
        queries: dict[str, str],
        corpus: dict[str, str],
        retrieval_results: dict[str, list[tuple[str, float]]],
        pseudo_labels: dict[str, dict[str, float]],
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Build feature matrix using pseudo-labels from click feedback.
        """
        X, y, groups = [], [], []

        for qid, query_text in queries.items():
            if qid not in retrieval_results or qid not in pseudo_labels:
                continue

            candidates = retrieval_results[qid]
            q_labels = pseudo_labels[qid]
            group_size = 0

            for rank, (pid, dense_score) in enumerate(candidates, start=1):
                if pid not in corpus:
                    continue

                features = compute_features(
                    query=query_text,
                    passage=corpus[pid],
                    dense_score=dense_score,
                    bm25_score=0.0,
                    retrieval_rank=rank,
                )

                X.append(features.to_array())
                y.append(q_labels.get(pid, 0.0))
                group_size += 1

            if group_size > 0:
                groups.append(group_size)

        return np.array(X), np.array(y), np.array(groups)