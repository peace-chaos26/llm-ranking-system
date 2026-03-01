# scripts/run_feedback.py
"""
Feedback simulation and retraining loop.

Experiment design:
1. Take current pipeline rankings (from M5)
2. Simulate 50 user sessions per query with position bias
3. Train two new rankers:
   a. Naive ranker: clicks = relevance (biased)
   b. IPS ranker:   clicks debiased by propensity weighting
4. Evaluate all three rankers:
   - Original LambdaMART (trained on ground truth labels)
   - Naive click ranker (biased)
   - IPS-debiased ranker
5. Show that IPS ranker recovers quality closer to ground truth
"""

import json
import os
import pickle
from pathlib import Path

import numpy as np
from loguru import logger

os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["OMP_NUM_THREADS"] = "1"

from src.evaluation.metrics import aggregate_metrics, compute_all_metrics
from src.feedback.click_simulator import ClickSimulator
from src.feedback.feedback_processor import FeedbackProcessor
from src.feedback.position_bias import PositionBiasModel
from src.ranking.lambdamart_ranker import LambdaMARTRanker
from src.utils.data_loader import load_corpus_as_dict, load_queries, load_qrels

MODEL_DIR = Path("experiments/models")
RESULTS_DIR = Path("experiments/results")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
MODEL_DIR.mkdir(parents=True, exist_ok=True)

N_QUERIES = 40
N_SESSIONS = 100       # simulated users per query
N_IMPRESSIONS = 10    # results shown per page


def evaluate_ranker(
    name: str,
    ranker: LambdaMARTRanker,
    queries: dict,
    retrieval_results: dict,
    qrels: dict,
    corpus: dict,
) -> dict:
    """Run ranker on all queries and compute NDCG/MRR."""
    per_query = []
    for qid, query_text in queries.items():
        if qid not in retrieval_results:
            continue
        reranked = ranker.rerank(
            query=query_text,
            candidates=retrieval_results[qid][:50],
            corpus=corpus,
        )
        retrieved = [pid for pid, _ in reranked[:10]]
        if qid in qrels:
            per_query.append(
                compute_all_metrics(retrieved, qrels[qid], k_values=[5, 10])
            )

    metrics = aggregate_metrics(per_query)
    logger.info(
        f"{name:<35} NDCG@10={metrics.get('ndcg@10', 0):.4f}  "
        f"MRR@10={metrics.get('mrr@10', 0):.4f}"
    )
    return metrics


def main():
    logger.info("=== Feedback Simulation & Retraining Loop ===")

    # Load data
    corpus = load_corpus_as_dict("data/processed/corpus.jsonl")
    queries = load_queries("data/processed/queries_sample.jsonl")
    qrels = load_qrels("data/processed/qrels_sample.tsv")

    with open(RESULTS_DIR / "retrieval_results_hybrid.pkl", "rb") as f:
        retrieval_results = pickle.load(f)

    # Build eval set
    valid_qids = [
        qid for qid in queries
        if qid in qrels and qid in retrieval_results
    ][:N_QUERIES]

    # Split: first 30 for training click models, last 10 for evaluation
    train_qids = valid_qids[:30]
    test_qids = valid_qids[30:]

    # Simulate clicks on TRAIN queries only
    eval_queries = {qid: queries[qid] for qid in train_qids}
    eval_qrels = {qid: qrels[qid] for qid in train_qids}
    eval_retrieval = {qid: retrieval_results[qid] for qid in train_qids}

    # Evaluate on TEST queries (unseen during click simulation)
    test_queries = {qid: queries[qid] for qid in test_qids}
    test_qrels = {qid: qrels[qid] for qid in test_qids}
    test_retrieval = {qid: retrieval_results[qid] for qid in test_qids}

    logger.info(f"Simulating feedback on {len(eval_queries)} queries")

    # ── Step 1: Load current pipeline rankings ────────────────────────────
    logger.info("\n[1/5] Loading original LambdaMART ranker...")
    original_ranker = LambdaMARTRanker(top_k_output=20)
    original_ranker.load(MODEL_DIR / "lambdamart.txt")

    # Get current rankings to simulate clicks on
    current_rankings = {}
    for qid, query_text in eval_queries.items():
        current_rankings[qid] = original_ranker.rerank(
            query=query_text,
            candidates=eval_retrieval[qid][:50],
            corpus=corpus,
        )

    # ── Step 2: Simulate clicks with position bias ────────────────────────
    logger.info(f"\n[2/5] Simulating {N_SESSIONS} user sessions per query...")
    bias_model = PositionBiasModel(eta=1.0)
    simulator = ClickSimulator(
        position_bias=bias_model,
        click_prob_relevant=0.8,
        click_prob_irrelevant=0.1,
    )

    all_events = simulator.simulate_batch(
        queries=eval_queries,
        ranked_results=current_rankings,
        qrels=eval_qrels,
        n_sessions_per_query=N_SESSIONS,
        n_impressions=N_IMPRESSIONS,
    )

    total_clicks = sum(1 for e in all_events if e.was_clicked)
    total_impressions = len(all_events)
    ctr = total_clicks / total_impressions if total_impressions > 0 else 0

    logger.info(f"Generated {total_impressions:,} impressions")
    logger.info(f"Total clicks: {total_clicks:,} (CTR={ctr:.3f})")

    # Show position bias effect
    logger.info("\nClick rate by position (demonstrates position bias):")
    for rank in range(1, N_IMPRESSIONS + 1):
        rank_events = [e for e in all_events if e.rank == rank]
        rank_clicks = sum(1 for e in rank_events if e.was_clicked)
        rank_ctr = rank_clicks / len(rank_events) if rank_events else 0
        bar = "█" * int(rank_ctr * 40)
        logger.info(f"  Rank {rank:2d}: {rank_ctr:.3f} {bar}")

    # ── Step 3: Build naive labels (biased) ──────────────────────────────
    logger.info("\n[3/5] Building naive click labels (biased)...")
    naive_processor = FeedbackProcessor(use_ips=False)
    naive_labels = naive_processor.clicks_to_labels(all_events)

    X_naive, y_naive, g_naive = naive_processor.build_training_features(
        eval_queries, corpus, eval_retrieval, naive_labels
    )

    # ── Step 4: Build IPS-debiased labels ────────────────────────────────
    logger.info("\n[4/5] Building IPS-debiased labels...")
    ips_processor = FeedbackProcessor(use_ips=True)
    ips_labels = ips_processor.clicks_to_labels(all_events)

    X_ips, y_ips, g_ips = ips_processor.build_training_features(
        eval_queries, corpus, eval_retrieval, ips_labels
    )

    # ── Step 5: Retrain and compare ───────────────────────────────────────
    logger.info("\n[5/5] Retraining rankers and comparing...")

    # Naive ranker
    naive_ranker = LambdaMARTRanker(top_k_output=20)
    naive_ranker.train(X_naive, y_naive, g_naive)
    naive_ranker.save(MODEL_DIR / "lambdamart_naive_clicks.txt")

    # IPS ranker
    ips_ranker = LambdaMARTRanker(top_k_output=20)
    ips_ranker.train(X_ips, y_ips, g_ips)
    ips_ranker.save(MODEL_DIR / "lambdamart_ips_clicks.txt")

    # ── Evaluation Comparison ─────────────────────────────────────────────
    logger.info("\n" + "=" * 70)
    logger.info("FEEDBACK EXPERIMENT RESULTS")
    logger.info("=" * 70)
    logger.info(f"{'Ranker':<35} {'NDCG@10':<12} {'MRR@10'}")
    logger.info("-" * 70)

    m_original = evaluate_ranker(
        "Original (ground truth labels)",
        original_ranker, test_queries, test_retrieval, test_qrels, corpus
    )
    m_naive = evaluate_ranker(
        "Naive clicks (biased)",
        naive_ranker, test_queries, test_retrieval, test_qrels, corpus
    )
    m_ips = evaluate_ranker(
        "IPS-debiased clicks",
        ips_ranker, test_queries, test_retrieval, test_qrels, corpus
    )

    logger.info("=" * 70)
    logger.info("\nKey insight:")
    ndcg_gap_naive = m_original["ndcg@10"] - m_naive["ndcg@10"]
    ndcg_gap_ips = m_original["ndcg@10"] - m_ips["ndcg@10"]
    if ndcg_gap_naive > 0:
        logger.info(f"  Naive ranker degrades NDCG by {ndcg_gap_naive:.4f} vs ground truth")
    else:
        logger.info(f"  Naive ranker improves NDCG by {abs(ndcg_gap_naive):.4f} (small test set, high variance)")
    logger.info(f"  IPS ranker gap vs ground truth: {ndcg_gap_ips:.4f}")
    recovery = ((ndcg_gap_naive - ndcg_gap_ips) / max(abs(ndcg_gap_naive), 1e-6)) * 100
    logger.info(f"  IPS vs Naive difference: {m_ips['ndcg@10'] - m_naive['ndcg@10']:.4f} NDCG")
    logger.info(f"  IPS recovered {((ndcg_gap_naive - ndcg_gap_ips) / max(ndcg_gap_naive, 1e-6)) * 100:.1f}% of the bias-induced degradation")

    # Save results
    results = {
        "simulation_config": {
            "n_queries": N_QUERIES,
            "n_sessions_per_query": N_SESSIONS,
            "n_impressions": N_IMPRESSIONS,
            "position_bias_eta": 1.0,
            "click_prob_relevant": 0.8,
            "click_prob_irrelevant": 0.1,
        },
        "click_stats": {
            "total_impressions": total_impressions,
            "total_clicks": total_clicks,
            "overall_ctr": round(ctr, 4),
        },
        "ranker_comparison": {
            "original_ground_truth": m_original,
            "naive_clicks_biased": m_naive,
            "ips_debiased": m_ips,
        },
    }

    with open(RESULTS_DIR / "feedback_results.json", "w") as f:
        json.dump(results, f, indent=2)
    logger.info(f"\nResults saved to {RESULTS_DIR}/feedback_results.json")


if __name__ == "__main__":
    main()