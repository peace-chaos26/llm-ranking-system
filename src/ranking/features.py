# src/ranking/features.py
"""
Feature engineering for Learning-to-Rank.

LambdaMART is a gradient boosted tree model — it needs hand-crafted
features for each (query, passage) pair. This is fundamentally different from neural rankers which learn features automatically.

Features we compute:
- Lexical overlap (BM25-style signals)
- Embedding similarity (from retrieval)
- Passage statistics (length, position)
"""

import re
import numpy as np
from dataclasses import dataclass


@dataclass
class RankingFeatures:
    """All features for a single (query, passage) pair."""
    
    # Lexical features
    exact_match_ratio: float        # fraction of query terms in passage
    query_term_coverage: float      # fraction of unique query terms found
    passage_length_words: int       # raw passage length
    passage_length_norm: float      # length normalized to [0,1]
    query_length: int               # query length in words
    
    # Retrieval features (from Stage 1)
    dense_score: float              # cosine similarity from FAISS
    bm25_score: float               # BM25 score
    retrieval_rank: int             # rank position in retrieval results (1-indexed)
    reciprocal_rank: float          # 1 / retrieval_rank
    
    # Text statistics
    avg_word_length: float          # proxy for vocabulary complexity
    digit_ratio: float              # fraction of tokens that are numbers
    uppercase_ratio: float          # fraction of uppercase words

    def to_array(self) -> np.ndarray:
        """Convert to numpy array for model input."""
        return np.array([
            self.exact_match_ratio,
            self.query_term_coverage,
            self.passage_length_words,
            self.passage_length_norm,
            self.query_length,
            self.dense_score,
            self.bm25_score,
            self.retrieval_rank,
            self.reciprocal_rank,
            self.avg_word_length,
            self.digit_ratio,
            self.uppercase_ratio,
        ], dtype=np.float32)

    @staticmethod
    def feature_names() -> list[str]:
        return [
            "exact_match_ratio",
            "query_term_coverage",
            "passage_length_words",
            "passage_length_norm",
            "query_length",
            "dense_score",
            "bm25_score",
            "retrieval_rank",
            "reciprocal_rank",
            "avg_word_length",
            "digit_ratio",
            "uppercase_ratio",
        ]


def compute_features(
    query: str,
    passage: str,
    dense_score: float,
    bm25_score: float,
    retrieval_rank: int,
    max_passage_length: int = 200,
) -> RankingFeatures:
    """
    Compute all ranking features for a (query, passage) pair.
    Designed to be fast — runs on 100 candidates per query.
    """
    query_tokens = query.lower().split()
    passage_tokens = passage.lower().split()
    query_terms = set(query_tokens)
    passage_terms = set(passage_tokens)

    # Lexical overlap
    matched_terms = query_terms & passage_terms
    exact_match_ratio = len(matched_terms) / len(query_terms) if query_terms else 0.0
    query_term_coverage = len(matched_terms) / len(query_terms) if query_terms else 0.0

    # Length features
    passage_length = len(passage_tokens)
    passage_length_norm = min(passage_length / max_passage_length, 1.0)

    # Text statistics
    words = passage_tokens
    avg_word_length = np.mean([len(w) for w in words]) if words else 0.0
    digit_ratio = sum(1 for w in words if w.isdigit()) / len(words) if words else 0.0
    uppercase_ratio = sum(1 for w in passage.split() if w.isupper()) / len(words) if words else 0.0

    return RankingFeatures(
        exact_match_ratio=exact_match_ratio,
        query_term_coverage=query_term_coverage,
        passage_length_words=passage_length,
        passage_length_norm=passage_length_norm,
        query_length=len(query_tokens),
        dense_score=dense_score,
        bm25_score=bm25_score,
        retrieval_rank=retrieval_rank,
        reciprocal_rank=1.0 / retrieval_rank,
        avg_word_length=float(avg_word_length),
        digit_ratio=digit_ratio,
        uppercase_ratio=uppercase_ratio,
    )