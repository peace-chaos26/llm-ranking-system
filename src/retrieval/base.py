# src/retrieval/base.py
"""
Abstract base class for all retrievers.

Why define an interface?
BM25Retriever, DenseRetriever, HybridRetriever.
All of them must be swappable — the ranking pipeline shouldn't
care which retriever it's talking to. This is the dependency
inversion principle applied to ML systems.

"I defined a retriever interface so any stage
of the pipeline can swap retrieval strategies without changing
downstream code."
"""

from abc import ABC, abstractmethod


class BaseRetriever(ABC):

    @abstractmethod
    def index(self, corpus: dict[str, str]) -> None:
        """
        Build the retrieval index from a corpus.
        corpus: {passage_id: passage_text}
        """
        ...

    @abstractmethod
    def retrieve(self, query: str, top_k: int = 100) -> list[tuple[str, float]]:
        """
        Retrieve top-k passages for a query.
        Returns: [(passage_id, score), ...] sorted by score descending
        """
        ...

    def batch_retrieve(
        self, queries: dict[str, str], top_k: int = 100
    ) -> dict[str, list[tuple[str, float]]]:
        """
        Retrieve for multiple queries.
        Returns: {query_id: [(passage_id, score), ...]}
        Default implementation loops over retrieve() — override for speed.
        """
        results = {}
        for qid, query_text in queries.items():
            results[qid] = self.retrieve(query_text, top_k=top_k)
        return results