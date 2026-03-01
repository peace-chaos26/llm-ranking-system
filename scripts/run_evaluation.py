# scripts/run_evaluation.py
"""
Full ablation evaluation across all pipeline configurations.

Ablation table produced:
  Config 1: Retrieval only          — baseline
  Config 2: + LambdaMART            — does Stage 2 help?
  Config 3: + LLM re-ranker         — does Stage 3 help?
  Config 4: + MMR diversity         — does diversity help?
"""
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["OMP_NUM_THREADS"] = "1"

import json
import os
import pickle
import time
from pathlib import Path

import numpy as np
from dotenv import load_dotenv
from loguru import logger

from src.evaluation.metrics import aggregate_metrics, compute_all_metrics
from src.ranking.diversity import MMRDiversifier
from src.ranking.lambdamart_ranker import LambdaMARTRanker
from src.ranking.llm_reranker import LLMReranker
from src.retrieval.bm25_retriever import BM25Retriever
from src.retrieval.dense_retriever import DenseRetriever
from src.retrieval.hybrid_retriever import HybridRetriever
from src.utils.data_loader import load_corpus_as_dict, load_queries, load_qrels

load_dotenv()

INDEX_DIR = Path("data/indexes")
MODEL_DIR = Path("experiments/models")
RESULTS_DIR = Path("experiments/results")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# Query limits per stage
N_EVAL = 20       # configs 1 + 2
N_LLM = 10        # config 3 — LLM is expensive, 10 is enough to show the gain


# ── Helpers ──────────────────────────────────────────────────────────────────

def evaluate(
    results: dict[str, list[tuple[str, float]]],
    qrels: dict[str, dict[str, int]],
    k_values: list[int] = [5, 10],
) -> dict[str, float]:
    """Compute aggregate metrics over all queries."""
    per_query = []
    for qid, ranked in results.items():
        if qid not in qrels:
            continue
        retrieved = [pid for pid, _ in ranked]
        per_query.append(compute_all_metrics(retrieved, qrels[qid], k_values))
    return aggregate_metrics(per_query)


def print_row(config: str, metrics: dict, latency: float, cost: float = 0.0):
    logger.info(
        f"{config:<35} "
        f"NDCG@10={metrics.get('ndcg@10', 0):.4f}  "
        f"MRR@10={metrics.get('mrr@10', 0):.4f}  "
        f"R@10={metrics.get('recall@10', 0):.4f}  "
        f"Latency={latency:.0f}ms  "
        f"Cost/1k=${cost:.2f}"
    )


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    logger.info("=== Loading data ===")
    corpus = load_corpus_as_dict("data/processed/corpus.jsonl")
    queries = load_queries("data/processed/queries_sample.jsonl")
    qrels = load_qrels("data/processed/qrels_sample.tsv")

    # Build eval set — queries that have qrels AND retrieval results
    with open(RESULTS_DIR / "retrieval_results_hybrid.pkl", "rb") as f:
        retrieval_results = pickle.load(f)

    valid_qids = [
        qid for qid in queries
        if qid in qrels and qid in retrieval_results
    ]
    logger.info(f"Valid eval queries: {len(valid_qids)}")

    # Hard cap
    eval_qids = valid_qids[:N_EVAL]
    eval_queries = {qid: queries[qid] for qid in eval_qids}
    eval_qrels = {qid: qrels[qid] for qid in eval_qids}

    logger.info(f"Running evaluation on {len(eval_qids)} queries")

    reports = []

    # ── Config 1: Retrieval Only ──────────────────────────────────────────
    logger.info("\n[1/4] Retrieval only...")

    t0 = time.perf_counter()

    # Re-run retrieval on eval queries for accurate latency measurement
    bm25 = BM25Retriever()
    bm25.load(INDEX_DIR / "bm25.pkl")
    dense = DenseRetriever(index_type="IVF", nprobe=32)
    dense.load(INDEX_DIR / "faiss_ivf.index", INDEX_DIR / "passage_ids.json")
    hybrid = HybridRetriever(dense_retriever=dense, sparse_retriever=bm25)
    retrieval_out = hybrid.batch_retrieve(eval_queries, top_k=100)

    retrieval_latency = ((time.perf_counter() - t0) / len(eval_qids)) * 1000

    m1 = evaluate(retrieval_out, eval_qrels)
    r1 = {"config": "1_retrieval_only", "metrics": m1,
          "latency_ms": round(retrieval_latency, 1), "cost_per_1k_usd": 0.0}
    reports.append(r1)
    print_row("1. Retrieval only", m1, retrieval_latency)

    # ── Config 2: + LambdaMART ────────────────────────────────────────────
    logger.info("\n[2/4] + LambdaMART (Stage 2)...")

    lambdamart = LambdaMARTRanker(top_k_output=20)
    lambdamart.load(MODEL_DIR / "lambdamart.txt")

    stage2_out = {}
    t0 = time.perf_counter()
    for qid, query_text in eval_queries.items():
        stage2_out[qid] = lambdamart.rerank(
            query=query_text,
            candidates=retrieval_out[qid][:50],
            corpus=corpus,
        )
    stage2_latency = retrieval_latency + ((time.perf_counter() - t0) / len(eval_qids)) * 1000

    m2 = evaluate(stage2_out, eval_qrels)
    r2 = {"config": "2_retrieval_lambdamart", "metrics": m2,
          "latency_ms": round(stage2_latency, 1), "cost_per_1k_usd": 0.0}
    reports.append(r2)
    print_row("2. + LambdaMART", m2, stage2_latency)

    # ── Config 3: + LLM Re-ranker ─────────────────────────────────────────
    logger.info(f"\n[3/4] + LLM Re-ranker (Stage 3) on {N_LLM} queries...")

    openai_key = os.getenv("OPENAI_API_KEY")
    if not openai_key:
        logger.warning("OPENAI_API_KEY not set — skipping LLM stage")
        stage3_out = {qid: results[:10] for qid, results in stage2_out.items()}
        stage3_latency = stage2_latency
        cost_per_1k = 0.0
    else:
        # Only run LLM on first N_LLM queries to control cost
        llm_qids = eval_qids[:N_LLM]
        llm_queries = {qid: eval_queries[qid] for qid in llm_qids}
        llm_qrels = {qid: eval_qrels[qid] for qid in llm_qids}

        llm_reranker = LLMReranker(model="gpt-4o-mini", top_k_output=10)

        stage3_out = {}
        t0 = time.perf_counter()
        for i, (qid, query_text) in enumerate(llm_queries.items()):
            logger.info(f"  LLM re-ranking query {i+1}/{len(llm_queries)}: '{query_text[:50]}'")
            stage3_out[qid] = llm_reranker.rerank(
                query=query_text,
                candidates=stage2_out[qid],
                corpus=corpus,
            )
        llm_elapsed = (time.perf_counter() - t0) / len(llm_queries) * 1000
        stage3_latency = stage2_latency + llm_elapsed

        cost_report = llm_reranker.get_cost_report()
        cost_per_1k = cost_report["cost_per_1k_requests_usd"]

        logger.info(f"LLM cost: ${cost_report['total_cost_usd']:.4f} "
                    f"for {cost_report['num_calls']} calls "
                    f"(${cost_per_1k:.2f}/1k requests)")

        m3 = evaluate(stage3_out, llm_qrels)
        r3 = {"config": "3_retrieval_lm_llm", "metrics": m3,
              "latency_ms": round(stage3_latency, 1), "cost_per_1k_usd": cost_per_1k}
        reports.append(r3)
        print_row("3. + LLM Re-ranker", m3, stage3_latency, cost_per_1k)

    # ── Config 4: + MMR Diversity ─────────────────────────────────────────
    logger.info("\n[4/4] + MMR Diversity (Stage 4)...")
    logger.info("Note: MMR loads sentence-transformers — takes ~30 seconds first run")

    diversifier = MMRDiversifier(lambda_param=0.5)

    # Run MMR on whichever queries made it through stage 3
    input_for_mmr = stage3_out if stage3_out else {
        qid: results[:10] for qid, results in stage2_out.items()
    }
    mmr_qrels = {qid: eval_qrels[qid] for qid in input_for_mmr if qid in eval_qrels}

    final_out = {}
    t0 = time.perf_counter()
    for qid, query_text in eval_queries.items():
        if qid not in input_for_mmr:
            continue
        final_out[qid] = diversifier.diversify(
            query=query_text,
            candidates=input_for_mmr[qid],
            corpus=corpus,
            top_k=10,
        )
    mmr_latency = stage3_latency + ((time.perf_counter() - t0) / max(len(final_out), 1)) * 1000

    m4 = evaluate(final_out, mmr_qrels)
    r4 = {"config": "4_full_pipeline", "metrics": m4,
          "latency_ms": round(mmr_latency, 1), "cost_per_1k_usd": cost_per_1k}
    reports.append(r4)
    print_row("4. + MMR Diversity", m4, mmr_latency, cost_per_1k)

    # ── Ablation Table ────────────────────────────────────────────────────
    logger.info("\n\n" + "=" * 90)
    logger.info("ABLATION STUDY")
    logger.info("=" * 90)
    logger.info(f"{'Configuration':<35} {'NDCG@10':<10} {'MRR@10':<10} {'R@10':<8} {'Latency':<12} {'Cost/1k'}")
    logger.info("-" * 90)
    for r in reports:
        m = r["metrics"]
        logger.info(
            f"{r['config']:<35} "
            f"{m.get('ndcg@10', 0):<10} "
            f"{m.get('mrr@10', 0):<10} "
            f"{m.get('recall@10', 0):<8} "
            f"{str(r['latency_ms'])+'ms':<12} "
            f"${r['cost_per_1k_usd']}"
        )
    logger.info("=" * 90)

    # Save
    output = {"reports": reports}
    with open(RESULTS_DIR / "ablation_results.json", "w") as f:
        json.dump(output, f, indent=2)
    logger.info(f"\nSaved to {RESULTS_DIR}/ablation_results.json")


if __name__ == "__main__":
    main()