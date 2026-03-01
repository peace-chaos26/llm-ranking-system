# src/ranking/pipeline.py
"""
Full ranking pipeline orchestrator.

Wires together all stages:
  Retrieval results → Stage2 (LambdaMART) → Stage3 (LLM) → MMR → Output

Design principle: each stage is independently skippable via config.
This lets you run ablation experiments:
  - Retrieval only
  - Retrieval + Stage2
  - Retrieval + Stage2 + Stage3
  - Full pipeline
"""

import time
from dataclasses import dataclass, field
from loguru import logger

from src.ranking.lambdamart_ranker import LambdaMARTRanker
from src.ranking.llm_reranker import LLMReranker
from src.ranking.diversity import MMRDiversifier


@dataclass
class PipelineResult:
    query_id: str
    query_text: str
    final_ranking: list[tuple[str, float]]      # [(passage_id, score), ...]
    stage2_ranking: list[tuple[str, float]]     # after LambdaMART
    stage3_ranking: list[tuple[str, float]]     # after LLM re-ranker
    latency_ms: dict[str, float] = field(default_factory=dict)


class RankingPipeline:

    def __init__(
        self,
        lambdamart: LambdaMARTRanker,
        llm_reranker: LLMReranker | None = None,
        diversifier: MMRDiversifier | None = None,
        use_llm: bool = True,
        use_diversity: bool = True,
    ):
        self.lambdamart = lambdamart
        self.llm_reranker = llm_reranker
        self.diversifier = diversifier
        self.use_llm = use_llm and llm_reranker is not None
        self.use_diversity = use_diversity and diversifier is not None

    def run(
        self,
        query_id: str,
        query_text: str,
        retrieval_results: list[tuple[str, float]],
        corpus: dict[str, str],
    ) -> PipelineResult:
        latency = {}

        # Stage 2: LambdaMART
        t0 = time.time()
        stage2 = self.lambdamart.rerank(
            query=query_text,
            candidates=retrieval_results,
            corpus=corpus,
        )
        latency["stage2_ms"] = round((time.time() - t0) * 1000, 2)

        # Stage 3: LLM Re-ranker (optional)
        if self.use_llm:
            t0 = time.time()
            stage3 = self.llm_reranker.rerank(
                query=query_text,
                candidates=stage2,
                corpus=corpus,
            )
            latency["stage3_ms"] = round((time.time() - t0) * 1000, 2)
        else:
            stage3 = stage2[:10]

        # Post-processing: MMR Diversity (optional)
        if self.use_diversity:
            t0 = time.time()
            final = self.diversifier.diversify(
                query=query_text,
                candidates=stage3,
                corpus=corpus,
                top_k=10,
            )
            latency["mmr_ms"] = round((time.time() - t0) * 1000, 2)
        else:
            final = stage3

        return PipelineResult(
            query_id=query_id,
            query_text=query_text,
            final_ranking=final,
            stage2_ranking=stage2,
            stage3_ranking=stage3,
            latency_ms=latency,
        )

    def batch_run(
        self,
        queries: dict[str, str],
        retrieval_results: dict[str, list[tuple[str, float]]],
        corpus: dict[str, str],
        max_queries: int | None = None,
    ) -> dict[str, PipelineResult]:
        results = {}
        query_items = list(queries.items())

        if max_queries:
            query_items = query_items[:max_queries]

        for qid, query_text in query_items:
            if qid not in retrieval_results:
                continue
            try:
                results[qid] = self.run(
                    query_id=qid,
                    query_text=query_text,
                    retrieval_results=retrieval_results[qid],
                    corpus=corpus,
                )
            except Exception as e:
                logger.warning(f"Pipeline failed for query {qid}: {e}")

        return results