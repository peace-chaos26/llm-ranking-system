# src/utils/data_loader.py

import json
from pathlib import Path
from typing import Generator

from loguru import logger


def load_corpus(path: str | Path) -> Generator[dict, None, None]:
    """
    Stream corpus passages one at a time.
    
    Yields: {"id": str, "text": str}
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Corpus not found at {path}. Run scripts/download_data.py first.")
    
    with open(path) as f:
        for line in f:
            yield json.loads(line.strip())


def load_corpus_as_dict(path: str | Path) -> dict[str, str]:
    """
    Load full corpus into memory as {passage_id: text}.
    Use only when you need random access (e.g. building indexes).
    """
    logger.info(f"Loading corpus into memory from {path}...")
    corpus = {doc["id"]: doc["text"] for doc in load_corpus(path)}
    logger.info(f"Loaded {len(corpus):,} passages")
    return corpus


def load_queries(path: str | Path) -> dict[str, str]:
    """Load queries as {query_id: query_text}."""
    path = Path(path)
    queries = {}
    with open(path) as f:
        for line in f:
            obj = json.loads(line.strip())
            queries[obj["id"]] = obj["text"]
    logger.info(f"Loaded {len(queries):,} queries from {path}")
    return queries


def load_qrels(path: str | Path) -> dict[str, dict[str, int]]:
    """
    Load relevance judgments from TREC format file.
    
    Returns: {query_id: {passage_id: relevance_score}}
    """
    path = Path(path)
    qrels = {}
    with open(path) as f:
        for line in f:
            parts = line.strip().split("\t")
            qid, _, pid, rel = parts
            if qid not in qrels:
                qrels[qid] = {}
            qrels[qid][pid] = int(rel)
    logger.info(f"Loaded qrels for {len(qrels):,} queries from {path}")
    return qrels