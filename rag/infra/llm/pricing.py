"""OpenAI model pricing table and cost estimation.

Token counts are captured throughout this project (embedding, generation,
Vision transcription), but a token count alone doesn't answer "how much did
this cost?" — this module converts token counts into a dollar estimate using
published per-model rates.

Prices are per 1M tokens, verified against OpenAI's published pricing as of
2026-07. Update ``MODEL_PRICING_PER_1M`` if OpenAI changes rates; this is a
static table, not a live lookup.
"""

from __future__ import annotations

# {model_id: {"input": $ per 1M input tokens, "output": $ per 1M output tokens}}
# Embedding models have no output tokens, so "output" is 0.0 for them.
MODEL_PRICING_PER_1M: dict[str, dict[str, float]] = {
    "gpt-4o-mini": {"input": 0.15, "output": 0.60},
    "gpt-4.1-mini": {"input": 0.40, "output": 1.60},
    "gpt-4.1-nano": {"input": 0.10, "output": 0.40},
    "text-embedding-3-small": {"input": 0.02, "output": 0.0},
}


def estimate_cost_usd(
    model: str, prompt_tokens: int, completion_tokens: int = 0
) -> float:
    """Estimate the dollar cost of a completed API call.

    Args:
        model: Model id as used in the API call (e.g. ``"gpt-4o-mini"``).
        prompt_tokens: Input/prompt tokens consumed.
        completion_tokens: Output/completion tokens consumed (0 for
            embedding calls, which have no output tokens).

    Returns:
        Estimated cost in USD. Returns 0.0 for a model not present in
        ``MODEL_PRICING_PER_1M`` — logged nowhere, since an unknown model is
        expected for local/free providers (e.g. the ``multilingual``
        embedding provider) which have no per-token dollar cost at all.
    """
    rates = MODEL_PRICING_PER_1M.get(model)
    if rates is None:
        return 0.0
    return (
        prompt_tokens * rates["input"] + completion_tokens * rates["output"]
    ) / 1_000_000
