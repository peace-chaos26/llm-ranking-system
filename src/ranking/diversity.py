# src/ranking/diversity.py
"""
Maximal Marginal Relevance (MMR) for result diversification.

Problem MMR solves:
Without diversity, a ranking system returns the top-10 most relevant passages,
which are often near-duplicates of each other. If passage 1 answers the query
perfectly, passages 2-5 might be slight paraphrases of passage 1.
The user gets redundant information.

MMR selects documents that are:
1. Relevant to the query (high similarity to query)
2. Different from already-selected documents (low similarity to selected set)

MMR formula:
  MMR(d) = λ * relevance(d, query) - (1-λ) * max_similarity(d, selected)

  λ = 1: pure relevance (no diversity)
  λ = 0: pure diversity (no relevance)
  λ = 0.5: balanced (our default)

Why MMR over other diversity methods?
- Simple and interpretable
- One tunable parameter (λ)
- Linear time complexity
- Works with any similarity function
- Used in production at LinkedIn, Pinterest for feed diversity
"""

import numpy as np
from sentence_transformers import SentenceTransformer
from loguru import logger


class MMRDiversifier:

    def __init__(
        self,
        model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        lambda_param: float = 0.5,
    ):
        self.lambda_param = lambda_param
        logger.info(f"Loading MMR embedding model: {model_name}")
        self.model = SentenceTransformer(model_name)

    def _cosine_sim(self, a: np.ndarray, b: np.ndarray) -> float:
        return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-9))

    def diversify(
        self,
        query: str,
        candidates: list[tuple[str, float]],
        corpus: dict[str, str],
        top_k: int = 10,
    ) -> list[tuple[str, float]]:
        """
        Apply MMR to select diverse, relevant results.
        
        candidates: [(passage_id, relevance_score), ...] from LLM re-ranker
        Returns: top_k passages selected by MMR
        """
        if len(candidates) <= top_k:
            return candidates

        # Encode query and all candidate passages
        texts = [corpus[pid] for pid, _ in candidates if pid in corpus]
        pids = [pid for pid, _ in candidates if pid in corpus]
        scores = {pid: score for pid, score in candidates if pid in corpus}

        if not texts:
            return candidates[:top_k]

        all_texts = [query] + texts
        embeddings = self.model.encode(all_texts, normalize_embeddings=True)

        query_emb = embeddings[0]
        doc_embs = embeddings[1:]

        # Normalize relevance scores to [0,1]
        raw_scores = np.array([scores[pid] for pid in pids])
        score_range = raw_scores.max() - raw_scores.min()
        if score_range > 0:
            norm_scores = (raw_scores - raw_scores.min()) / score_range
        else:
            norm_scores = np.ones(len(raw_scores))

        selected_indices = []
        remaining_indices = list(range(len(pids)))

        for _ in range(min(top_k, len(pids))):
            if not remaining_indices:
                break

            best_idx = None
            best_mmr = -np.inf

            for idx in remaining_indices:
                relevance = self.lambda_param * norm_scores[idx]

                if selected_indices:
                    max_sim = max(
                        self._cosine_sim(doc_embs[idx], doc_embs[sel])
                        for sel in selected_indices
                    )
                    redundancy = (1 - self.lambda_param) * max_sim
                else:
                    redundancy = 0.0

                mmr_score = relevance - redundancy

                if mmr_score > best_mmr:
                    best_mmr = mmr_score
                    best_idx = idx

            selected_indices.append(best_idx)
            remaining_indices.remove(best_idx)

        return [(pids[i], scores[pids[i]]) for i in selected_indices]