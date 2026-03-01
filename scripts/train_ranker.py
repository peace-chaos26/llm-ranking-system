# scripts/train_ranker.py
"""
Train the LambdaMART ranker on MS MARCO retrieval results.

Training data strategy:
We use the hybrid retrieval results as our candidate set.
For each query, candidates are labeled using qrels:
  - passage in qrels with relevance=1 → label=1
  - all other retrieved passages → label=0
"""

import pickle
from pathlib import Path
from sklearn.model_selection import train_test_split
from loguru import logger

from src.utils.data_loader import load_corpus_as_dict, load_queries, load_qrels
from src.ranking.lambdamart_ranker import LambdaMARTRanker

RESULTS_DIR = Path("experiments/results")
MODEL_DIR = Path("experiments/models")
MODEL_DIR.mkdir(parents=True, exist_ok=True)


def main():
    logger.info("=== Training LambdaMART Ranker ===")

    # Load data
    corpus = load_corpus_as_dict("data/processed/corpus.jsonl")
    queries = load_queries("data/processed/queries_sample.jsonl")
    qrels = load_qrels("data/processed/qrels_sample.tsv")

    # Load retrieval results saved from M3
    with open(RESULTS_DIR / "retrieval_results_hybrid.pkl", "rb") as f:
        retrieval_results = pickle.load(f)

    # Only train on queries that have both retrieval results and qrels
    valid_qids = list(set(retrieval_results.keys()) & set(qrels.keys()))
    logger.info(f"Training queries: {len(valid_qids)}")

    train_qids, val_qids = train_test_split(valid_qids, test_size=0.2, random_state=42)

    train_queries = {qid: queries[qid] for qid in train_qids if qid in queries}
    val_queries = {qid: queries[qid] for qid in val_qids if qid in queries}
    train_retrieval = {qid: retrieval_results[qid] for qid in train_qids}
    val_retrieval = {qid: retrieval_results[qid] for qid in val_qids}
    train_qrels = {qid: qrels[qid] for qid in train_qids}
    val_qrels = {qid: qrels[qid] for qid in val_qids}

    ranker = LambdaMARTRanker(top_k_output=20)

    logger.info("Preparing training features...")
    X_train, y_train, groups_train = ranker.prepare_training_data(
        train_queries, corpus, train_retrieval, train_qrels
    )

    logger.info("Preparing validation features...")
    X_val, y_val, groups_val = ranker.prepare_training_data(
        val_queries, corpus, val_retrieval, val_qrels
    )

    logger.info(f"Train: {len(X_train)} pairs, Val: {len(X_val)} pairs")

    ranker.train(X_train, y_train, groups_train, X_val, y_val, groups_val)
    ranker.save(MODEL_DIR / "lambdamart.txt")

    logger.info("=== Training complete ===")


if __name__ == "__main__":
    main()