# src/retrieval/hybrid_retriever.py
"""
Hybrid retriever combining dense + sparse via Reciprocal Rank Fusion (RRF).

Why hybrid search?
Dense retrieval excels at semantic similarity ("what is the capital of France" 
finds passages about Paris even without exact word match).
BM25 excels at exact keyword matching (product codes, names, rare terms).
Neither dominates across all query types.

Reciprocal Rank Fusion:
  RRF_score(d) = Σ 1 / (k + rank(d))
  
  Where rank(d) is the position of document d in each ranked list,
  and k=60 is a smoothing constant (empirically tuned, rarely changed).

Why RRF over score normalization (e.g. linear combination)?
BM25 scores and cosine similarity scores live on completely different scales.
Normalizing them requires knowing the min/max of each, which changes with
every query. RRF only uses rank positions — no normalization needed.
It's robust, parameter-free, and empirically competitive with learned fusion.

"We chose RRF over learned fusion because it requires
no training data, is deterministic, and adds zero latency. A learned
fusion model is only worth it if you have click data to train on."
"""

from collections import defaultdict
from loguru import logger

from src.retrieval.base import BaseRetriever
from src.retrieval.bm25_retriever import BM25Retriever
from src.retrieval.dense_retriever import DenseRetriever


class HybridRetriever(BaseRetriever):

    def __init__(
        self,
        dense_retriever: DenseRetriever,
        sparse_retriever: BM25Retriever,
        dense_weight: float = 0.7,
        sparse_weight: float = 0.3,
        rrf_k: int = 60,
        retrieval_multiplier: int = 3,  # fetch top_k * multiplier from each, then fuse
    ):
        self.dense = dense_retriever
        self.sparse = sparse_retriever
        self.dense_weight = dense_weight
        self.sparse_weight = sparse_weight
        self.rrf_k = rrf_k
        self.retrieval_multiplier = retrieval_multiplier

    def index(self, corpus: dict[str, str]) -> None:
        """Index is built on the underlying retrievers separately."""
        raise NotImplementedError(
            "Call .dense.index() and .sparse.index() separately, "
            "then load them into this HybridRetriever."
        )

    def _rrf_fuse(
        self,
        dense_results: list[tuple[str, float]],
        sparse_results: list[tuple[str, float]],
        top_k: int,
    ) -> list[tuple[str, float]]:
        """
        Fuse two ranked lists using Reciprocal Rank Fusion.
        Returns top_k results sorted by fused score.
        """
        scores = defaultdict(float)

        for rank, (pid, _) in enumerate(dense_results):
            scores[pid] += self.dense_weight * (1.0 / (self.rrf_k + rank + 1))

        for rank, (pid, _) in enumerate(sparse_results):
            scores[pid] += self.sparse_weight * (1.0 / (self.rrf_k + rank + 1))

        fused = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        return fused[:top_k]

    def retrieve(self, query: str, top_k: int = 100) -> list[tuple[str, float]]:
        fetch_k = top_k * self.retrieval_multiplier

        dense_results = self.dense.retrieve(query, top_k=fetch_k)
        sparse_results = self.sparse.retrieve(query, top_k=fetch_k)

        return self._rrf_fuse(dense_results, sparse_results, top_k=top_k)

    def batch_retrieve(
        self, queries: dict[str, str], top_k: int = 100
    ) -> dict[str, list[tuple[str, float]]]:
        fetch_k = top_k * self.retrieval_multiplier

        dense_all = self.dense.batch_retrieve(queries, top_k=fetch_k)
        sparse_all = self.sparse.batch_retrieve(queries, top_k=fetch_k)

        results = {}
        for qid in queries:
            results[qid] = self._rrf_fuse(
                dense_all.get(qid, []),
                sparse_all.get(qid, []),
                top_k=top_k,
            )
        return results