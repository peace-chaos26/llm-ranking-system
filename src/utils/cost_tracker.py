# src/utils/cost_tracker.py
import tiktoken
from dataclasses import dataclass, field
from loguru import logger


COST_PER_1K_TOKENS = {
    "gpt-4o-mini": {"input": 0.00015, "output": 0.0006},
    "claude-haiku-3": {"input": 0.00025, "output": 0.00125},
}


@dataclass
class CostTracker:
    """
    Tracks LLM API token usage and cost across a full pipeline run.
    """
    model: str
    input_tokens: int = 0
    output_tokens: int = 0
    num_calls: int = 0
    _costs: dict = field(default_factory=dict, init=False)

    def __post_init__(self):
        if self.model not in COST_PER_1K_TOKENS:
            raise ValueError(f"Unknown model: {self.model}. Add it to COST_PER_1K_TOKENS.")
        self._costs = COST_PER_1K_TOKENS[self.model]

    def add_call(self, input_tokens: int, output_tokens: int) -> None:
        self.input_tokens += input_tokens
        self.output_tokens += output_tokens
        self.num_calls += 1

    @property
    def total_cost_usd(self) -> float:
        input_cost = (self.input_tokens / 1000) * self._costs["input"]
        output_cost = (self.output_tokens / 1000) * self._costs["output"]
        return round(input_cost + output_cost, 6)

    @property
    def cost_per_1k_requests(self) -> float:
        if self.num_calls == 0:
            return 0.0
        return round((self.total_cost_usd / self.num_calls) * 1000, 4)

    def report(self) -> dict:
        report = {
            "model": self.model,
            "num_calls": self.num_calls,
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "total_cost_usd": self.total_cost_usd,
            "cost_per_1k_requests_usd": self.cost_per_1k_requests,
        }
        logger.info("Cost report", **report)
        return report