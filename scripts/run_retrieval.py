# scripts/run_retrieval.py
"""
Benchmark all retrievers on the sample query set.
Outputs a comparison table of Recall@k and latency.

This script answers: which retriever should feed our ranking pipeline?
"""

import time
import json
from pathlib import Path

from loguru import logger
from src.utils.data_loader import load_queries, load_qrels
from src.retrieval.bm25_retriever import BM25Retriever
from src.retrieval.dense_retriever import DenseRetriever
from src.retrieval.hybrid_retriever import HybridRetriever

INDEX_DIR = Path("data/indexes")
RESULTS_DIR = Path("experiments/results")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


def recall_at_k(results: dict, qrels: dict, k: int) -> float:
    """
    Recall@k: fraction of relevant docs found in top-k.
    
    For MS MARCO (avg 1 relevant doc per query), Recall@k ≈ Hit Rate@k.
    "Did we find the one relevant document in our top-k candidates?"
    """
    hits = 0
    total_queries = 0

    for qid, retrieved in results.items():
        if qid not in qrels:
            continue
        relevant = set(qrels[qid].keys())
        retrieved_ids = {pid for pid, _ in retrieved[:k]}
        if relevant & retrieved_ids:
            hits += 1
        total_queries += 1

    return hits / total_queries if total_queries > 0 else 0.0


def benchmark_retriever(name, retriever, queries, qrels, top_k=100):
    logger.info(f"Benchmarking {name}...")

    start = time.time()
    results = retriever.batch_retrieve(queries, top_k=top_k)
    elapsed = time.time() - start

    latency_ms = (elapsed / len(queries)) * 1000

    metrics = {
        "retriever": name,
        "num_queries": len(queries),
        "recall@10": round(recall_at_k(results, qrels, 10), 4),
        "recall@20": round(recall_at_k(results, qrels, 20), 4),
        "recall@50": round(recall_at_k(results, qrels, 50), 4),
        "recall@100": round(recall_at_k(results, qrels, 100), 4),
        "avg_latency_ms": round(latency_ms, 2),
        "total_time_s": round(elapsed, 2),
    }

    logger.info(f"  Recall@10:  {metrics['recall@10']:.4f}")
    logger.info(f"  Recall@100: {metrics['recall@100']:.4f}")
    logger.info(f"  Latency:    {metrics['avg_latency_ms']:.1f}ms/query")

    return metrics, results


def main():
    queries = load_queries("data/processed/queries_sample.jsonl")
    qrels = load_qrels("data/processed/qrels_sample.tsv")

    # Only benchmark queries that have relevance labels
    eval_queries = {qid: q for qid, q in queries.items() if qid in qrels}
    logger.info(f"Evaluating on {len(eval_queries)} queries with relevance labels")

    # Load pre-built indexes
    bm25 = BM25Retriever()
    bm25.load(INDEX_DIR / "bm25.pkl")

    dense = DenseRetriever(index_type="IVF", nprobe=32)
    dense.load(INDEX_DIR / "faiss_ivf.index", INDEX_DIR / "passage_ids.json")

    hybrid = HybridRetriever(dense_retriever=dense, sparse_retriever=bm25)

    all_metrics = []

    bm25_metrics, _ = benchmark_retriever("BM25", bm25, eval_queries, qrels)
    dense_metrics, _ = benchmark_retriever("Dense-IVF", dense, eval_queries, qrels)
    hybrid_metrics, hybrid_results = benchmark_retriever("Hybrid-RRF", hybrid, eval_queries, qrels)

    all_metrics = [bm25_metrics, dense_metrics, hybrid_metrics]

    # Print comparison table
    logger.info("\n=== Retrieval Benchmark Results ===")
    logger.info(f"{'Retriever':<15} {'R@10':<8} {'R@20':<8} {'R@50':<8} {'R@100':<8} {'Latency'}")
    logger.info("-" * 65)
    for m in all_metrics:
        logger.info(
            f"{m['retriever']:<15} {m['recall@10']:<8} {m['recall@20']:<8} "
            f"{m['recall@50']:<8} {m['recall@100']:<8} {m['avg_latency_ms']:.1f}ms"
        )

    # Save results
    output_path = RESULTS_DIR / "retrieval_benchmark.json"
    with open(output_path, "w") as f:
        json.dump(all_metrics, f, indent=2)
    logger.info(f"Results saved to {output_path}")

    # Save hybrid results for M4 (ranker input)
    import pickle
    with open(RESULTS_DIR / "retrieval_results_hybrid.pkl", "wb") as f:
        pickle.dump(hybrid_results, f)
    logger.info("Hybrid retrieval results saved for ranking pipeline")


if __name__ == "__main__":
    main()