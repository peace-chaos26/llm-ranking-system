# scripts/validate_data.py

from pathlib import Path
from src.utils.data_loader import load_corpus_as_dict, load_queries, load_qrels
from loguru import logger


def validate(processed_dir: str = "data/processed"):
    p = Path(processed_dir)
    
    corpus = load_corpus_as_dict(p / "corpus.jsonl")
    queries = load_queries(p / "queries.jsonl")
    qrels = load_qrels(p / "qrels.tsv")

    logger.info("=== Validation Report ===")
    logger.info(f"Corpus size:         {len(corpus):,} passages")
    logger.info(f"Queries:             {len(queries):,}")
    logger.info(f"Queries with qrels:  {len(qrels):,}")

    # Check 1: every qrel query exists in queries
    missing_queries = set(qrels.keys()) - set(queries.keys())
    assert len(missing_queries) == 0, f"{len(missing_queries)} qrel queries missing from queries file"

    # Check 2: every relevant passage exists in corpus
    missing_passages = 0
    for qid, passages in qrels.items():
        for pid in passages:
            if pid not in corpus:
                missing_passages += 1
    assert missing_passages == 0, f"{missing_passages} relevant passages missing from corpus"

    # Check 3: no empty texts
    empty_passages = sum(1 for text in corpus.values() if not text.strip())
    assert empty_passages == 0, f"{empty_passages} empty passages in corpus"

    # Stats
    avg_passage_len = sum(len(t.split()) for t in corpus.values()) / len(corpus)
    avg_query_len = sum(len(t.split()) for t in queries.values()) / len(queries)
    avg_relevant = sum(len(v) for v in qrels.values()) / len(qrels)

    logger.info(f"Avg passage length:  {avg_passage_len:.1f} words")
    logger.info(f"Avg query length:    {avg_query_len:.1f} words")
    logger.info(f"Avg relevant docs/query: {avg_relevant:.2f}")
    logger.info("=== All checks passed ✓ ===")


if __name__ == "__main__":
    validate()