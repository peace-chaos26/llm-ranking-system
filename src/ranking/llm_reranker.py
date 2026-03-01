# src/ranking/llm_reranker.py
"""
Stage 3: LLM Re-ranker using GPT-4o-mini.

Why LLM re-ranking?
LambdaMART uses hand-crafted features and sees each (query, passage) 
pair independently. It can't reason about:
- Semantic nuance ("what's the best treatment" vs "what causes the disease")
- Multi-hop relevance
- Contextual coherence across passages

An LLM reads the query and ALL top-20 passages together and reasons
about their relative relevance holistically. This is expensive but
produces meaningfully better ranking for complex queries.

Prompting strategy: Pointwise scoring
We ask the LLM to score each passage 1-10 for relevance.
Alternative: Listwise (rank all at once) — more tokens, less stable.
Alternative: Pairwise (compare pairs) — O(n²) API calls, too expensive.
Pointwise is the best cost/quality tradeoff for production.

Cost control:
- Only run on top-20 from Stage 2 (not top-100)
- Truncate passages to 200 words
- Use cheapest capable model (gpt-4o-mini)
- Track every token with CostTracker
"""

import os
import json
import re
from loguru import logger
from openai import OpenAI

from src.utils.cost_tracker import CostTracker


RERANK_PROMPT = """You are a relevance scoring system. Your task is to score how relevant a passage is to a query.

Query: {query}

Passage: {passage}

Score the relevance of this passage to the query on a scale of 1-10:
- 10: Perfectly answers the query with specific, accurate information
- 7-9: Highly relevant, directly addresses the query
- 4-6: Partially relevant, touches on the topic but incomplete
- 1-3: Minimally relevant or off-topic

Respond with ONLY a JSON object: {{"score": <integer 1-10>, "reason": "<one sentence>"}}"""


class LLMReranker:

    def __init__(
        self,
        model: str = "gpt-4o-mini",
        top_k_output: int = 10,
        max_passage_words: int = 200,
    ):
        self.model = model
        self.top_k_output = top_k_output
        self.max_passage_words = max_passage_words
        self.client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        self.cost_tracker = CostTracker(model=model)

    def _truncate(self, text: str) -> str:
        words = text.split()
        return " ".join(words[:self.max_passage_words])

    def _score_passage(self, query: str, passage: str) -> tuple[float, str]:
        """
        Score a single (query, passage) pair using LLM.
        Returns (score, reason).
        Falls back to score=0 on any API error — never crash the pipeline.
        """
        prompt = RERANK_PROMPT.format(
            query=query,
            passage=self._truncate(passage),
        )

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=100,
                temperature=0.0,    # deterministic — ranking must be reproducible
            )

            self.cost_tracker.add_call(
                input_tokens=response.usage.prompt_tokens,
                output_tokens=response.usage.completion_tokens,
            )

            content = response.choices[0].message.content.strip()
            parsed = json.loads(content)
            return float(parsed["score"]), parsed.get("reason", "")

        except json.JSONDecodeError:
            # Try to extract score with regex as fallback
            match = re.search(r'"score"\s*:\s*(\d+)', content)
            if match:
                return float(match.group(1)), ""
            logger.warning(f"Failed to parse LLM response: {content}")
            return 0.0, "parse_error"

        except Exception as e:
            logger.warning(f"LLM API error: {e}")
            return 0.0, "api_error"

    def rerank(
        self,
        query: str,
        candidates: list[tuple[str, float]],
        corpus: dict[str, str],
    ) -> list[tuple[str, float]]:
        """
        Re-rank candidates using LLM relevance scoring.
        Input: top-20 from LambdaMART
        Output: top-10 re-ranked by LLM score
        """
        scored = []

        for pid, stage2_score in candidates:
            if pid not in corpus:
                continue

            llm_score, reason = self._score_passage(query, corpus[pid])
            scored.append((pid, llm_score))

            logger.debug(f"  pid={pid[:20]}... score={llm_score} reason={reason}")

        scored.sort(key=lambda x: x[1], reverse=True)
        return scored[:self.top_k_output]

    def get_cost_report(self) -> dict:
        return self.cost_tracker.report()