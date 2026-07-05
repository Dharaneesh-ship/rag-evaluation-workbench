"""Unit tests for the OpenAI model pricing table and cost estimation."""

import pytest

from rag.infra.llm.pricing import MODEL_PRICING_PER_1M, estimate_cost_usd


@pytest.mark.unit
class TestEstimateCostUsd:
    def test_gpt4o_mini_input_and_output(self):
        cost = estimate_cost_usd("gpt-4o-mini", prompt_tokens=1_000_000, completion_tokens=1_000_000)
        assert cost == pytest.approx(0.15 + 0.60)

    def test_embedding_model_has_no_output_cost(self):
        cost = estimate_cost_usd(
            "text-embedding-3-small", prompt_tokens=1_000_000, completion_tokens=1_000_000
        )
        # Embeddings have no completion tokens in practice; the rate table
        # reflects that by pricing "output" at 0 regardless of the value passed.
        assert cost == pytest.approx(0.02)

    def test_zero_tokens_is_zero_cost(self):
        assert estimate_cost_usd("gpt-4o-mini", 0, 0) == 0.0

    def test_unknown_model_returns_zero_not_an_error(self):
        # Local/free providers (e.g. the multilingual embedding provider)
        # have no per-token dollar cost and aren't in the pricing table.
        assert estimate_cost_usd("some-local-model", 10_000, 5_000) == 0.0

    def test_small_realistic_call(self):
        # A single Vision page transcription: ~1500 input tokens (image +
        # prompt), ~400 output tokens (markdown transcript).
        cost = estimate_cost_usd("gpt-4o-mini", prompt_tokens=1500, completion_tokens=400)
        expected = (1500 * 0.15 + 400 * 0.60) / 1_000_000
        assert cost == pytest.approx(expected)

    def test_all_pricing_table_entries_have_input_and_output_keys(self):
        for model, rates in MODEL_PRICING_PER_1M.items():
            assert "input" in rates, f"{model} missing 'input' rate"
            assert "output" in rates, f"{model} missing 'output' rate"
            assert rates["input"] >= 0
            assert rates["output"] >= 0
