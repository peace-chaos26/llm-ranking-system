# scripts/download_data.py

import json
import random
from pathlib import Path

from datasets import load_dataset
from loguru import logger
from tqdm import tqdm


DATA_DIR = Path("data")
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"

MAX_CORPUS_SIZE = 100_000
SEED = 42


def download_msmarco():
    """Download MS MARCO passage ranking dataset from HuggingFace."""
    logger.info("Downloading MS MARCO passage ranking dataset...")

    # MS MARCO on HuggingFace — 'ms_marco' with 'v2.1' config
    # This is the passage ranking version (not document ranking)
    dataset = load_dataset("ms_marco", "v2.1", trust_remote_code=True)
    logger.info("Download complete.")
    return dataset


def build_corpus(dataset, max_size: int = MAX_CORPUS_SIZE) -> dict:
    """
    Extract unique passages from the dataset to form our retrieval corpus.
    
    Returns: {passage_id: passage_text}
    """
    logger.info(f"Building corpus (max {max_size:,} passages)...")
    
    corpus = {}
    
    # Passages are nested inside each query's 'passages' field
    for split in ["train", "validation"]:
        for example in tqdm(dataset[split], desc=f"Processing {split}"):
            for passage_text, is_selected, passage_id in zip(
                example["passages"]["passage_text"],
                example["passages"]["is_selected"],
                example["passages"]["url"],   # url acts as passage id in v2.1
            ):
                if passage_text and passage_text not in corpus:
                    corpus[passage_id] = passage_text
                
                if len(corpus) >= max_size:
                    break
            
            if len(corpus) >= max_size:
                break
    
    logger.info(f"Corpus size: {len(corpus):,} passages")
    return corpus


def build_queries_and_qrels(dataset) -> tuple[dict, dict]:
    """
    Extract queries and relevance judgments (qrels).
    
    qrels format: {query_id: {passage_id: relevance_score}}
    relevance_score: 1 = relevant, 0 = not relevant
    
    We only keep queries that have at least one relevant passage
    in our corpus — otherwise evaluation metrics are undefined.
    """
    logger.info("Building queries and qrels...")
    
    queries = {}
    qrels = {}
    
    for split in ["validation"]:   # validation has clean eval queries
        for example in tqdm(dataset[split], desc=f"Processing {split}"):
            qid = str(example["query_id"])
            query_text = example["query"]
            
            relevant_passages = {}
            for passage_text, is_selected, passage_id in zip(
                example["passages"]["passage_text"],
                example["passages"]["is_selected"],
                example["passages"]["url"],
            ):
                if is_selected == 1:
                    relevant_passages[passage_id] = 1
            
            if relevant_passages:   # only keep queries with known relevant docs
                queries[qid] = query_text
                qrels[qid] = relevant_passages
    
    logger.info(f"Queries with relevance labels: {len(queries):,}")
    return queries, qrels


def save_corpus(corpus: dict, path: Path) -> None:
    """Save corpus as JSONL. One passage per line."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        for pid, text in tqdm(corpus.items(), desc="Saving corpus"):
            f.write(json.dumps({"id": pid, "text": text}) + "\n")
    logger.info(f"Corpus saved to {path} ({path.stat().st_size / 1e6:.1f} MB)")


def save_queries(queries: dict, path: Path) -> None:
    """Save queries as JSONL."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        for qid, text in queries.items():
            f.write(json.dumps({"id": qid, "text": text}) + "\n")
    logger.info(f"Queries saved to {path} ({len(queries):,} queries)")


def save_qrels(qrels: dict, path: Path) -> None:
    """
    Save qrels in TREC format: query_id 0 passage_id relevance
    
    Why TREC format?
    It's the standard format used by all IR evaluation tools
    (trec_eval, pytrec_eval). Using it means your evaluation
    code is compatible with the entire research ecosystem.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        for qid, passages in qrels.items():
            for pid, rel in passages.items():
                f.write(f"{qid}\t0\t{pid}\t{rel}\n")
    logger.info(f"Qrels saved to {path}")


def main():
    logger.info("=== MS MARCO Data Pipeline ===")
    
    dataset = download_msmarco()
    
    corpus = build_corpus(dataset, max_size=MAX_CORPUS_SIZE)
    queries, qrels = build_queries_and_qrels(dataset)
    
    # Filter qrels to only include passages that exist in our corpus
    filtered_qrels = {}
    for qid, passages in qrels.items():
        filtered = {pid: rel for pid, rel in passages.items() if pid in corpus}
        if filtered:
            filtered_qrels[qid] = filtered
    
    logger.info(f"Queries after corpus filtering: {len(filtered_qrels):,}")
    
    # Save everything
    save_corpus(corpus, PROCESSED_DIR / "corpus.jsonl")
    save_queries(queries, PROCESSED_DIR / "queries.jsonl")
    save_qrels(filtered_qrels, PROCESSED_DIR / "qrels.tsv")
    
    # Save a small sample for quick dev/testing iterations
    sample_queries = dict(list(filtered_qrels.items())[:500])
    save_queries(
        {qid: queries[qid] for qid in sample_queries},
        PROCESSED_DIR / "queries_sample.jsonl"
    )
    save_qrels(sample_queries, PROCESSED_DIR / "qrels_sample.tsv")
    
    logger.info("=== Data pipeline complete ===")
    logger.info(f"  Corpus:  {MAX_CORPUS_SIZE:,} passages")
    logger.info(f"  Queries: {len(filtered_qrels):,} with relevance labels")
    logger.info(f"  Sample:  500 queries for fast iteration")


if __name__ == "__main__":
    main()