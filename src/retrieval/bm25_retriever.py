# src/retrieval/bm25_retriever.py
"""
BM25 sparse retriever using rank-bm25 library.

What is BM25?
Best Match 25 — a probabilistic ranking function based on term frequency
and inverse document frequency.

BM25 score for a query q and document d:
  score(q,d) = Σ IDF(qi) * (f(qi,d) * (k1+1)) / (f(qi,d) + k1*(1 - b + b*|d|/avgdl))

Where:
  f(qi, d) = term frequency of query term qi in document d
  |d|       = document length
  avgdl     = average document length across corpus
  k1, b     = tuning parameters (defaults: k1=1.5, b=0.75)
"""

import pickle
from pathlib import Path

from rank_bm25 import BM25Okapi
from loguru import logger
from tqdm import tqdm

from src.retrieval.base import BaseRetriever


class BM25Retriever(BaseRetriever):

    def __init__(self):
        self.bm25 = None
        self.passage_ids: list[str] = []    # maps integer index -> passage_id
        self._indexed = False

    def index(self, corpus: dict[str, str]) -> None:
        """
        Tokenize all passages and build BM25 index.
        """
        logger.info(f"Building BM25 index over {len(corpus):,} passages...")

        self.passage_ids = list(corpus.keys())
        tokenized_corpus = [
            corpus[pid].lower().split()
            for pid in tqdm(self.passage_ids, desc="Tokenizing")
        ]

        self.bm25 = BM25Okapi(tokenized_corpus)
        self._indexed = True
        logger.info("BM25 index built.")

    def retrieve(self, query: str, top_k: int = 100) -> list[tuple[str, float]]:
        """Retrieve top-k passages for a query using BM25 scoring."""
        if not self._indexed:
            raise RuntimeError("Call index() before retrieve()")

        tokenized_query = query.lower().split()
        scores = self.bm25.get_scores(tokenized_query)

        # Get top-k indices sorted by score descending
        import numpy as np
        top_indices = np.argsort(scores)[::-1][:top_k]

        return [
            (self.passage_ids[i], float(scores[i]))
            for i in top_indices
            if scores[i] > 0   # BM25 score of 0 means no term overlap
        ]

    def save(self, path: str | Path) -> None:
        """Persist index to disk. BM25 index build takes ~2 min — save it."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump({"bm25": self.bm25, "passage_ids": self.passage_ids}, f)
        logger.info(f"BM25 index saved to {path}")

    def load(self, path: str | Path) -> None:
        """Load persisted index."""
        with open(Path(path), "rb") as f:
            data = pickle.load(f)
        self.bm25 = data["bm25"]
        self.passage_ids = data["passage_ids"]
        self._indexed = True
        logger.info(f"BM25 index loaded from {path}")