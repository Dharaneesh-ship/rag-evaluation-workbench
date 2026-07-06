"""Tests for OpenAIEmbeddingProvider — all API calls are mocked."""

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from rag.infra.embedding.openai_embedding import OpenAIEmbeddingProvider


def _make_response(vectors: list[list[float]], prompt_tokens: int = 10):
    """Build a fake openai embeddings response."""
    data = [
        SimpleNamespace(index=i, embedding=v) for i, v in enumerate(vectors)
    ]
    usage = SimpleNamespace(prompt_tokens=prompt_tokens)
    return SimpleNamespace(data=data, usage=usage)


@pytest.fixture()
def provider():
    """Provider with a mocked OpenAI client."""
    with patch("rag.infra.embedding.openai_embedding.OpenAI") as MockClient:
        mock_client = MagicMock()
        MockClient.return_value = mock_client
        p = OpenAIEmbeddingProvider(model="text-embedding-3-small", dim=4)
        p._client = mock_client
        yield p, mock_client


def test_embed_returns_correct_count(provider):
    p, client = provider
    client.embeddings.create.return_value = _make_response(
        [[0.1, 0.2, 0.3, 0.4], [0.5, 0.6, 0.7, 0.8]]
    )

    vectors = p.embed(["hello", "world"])

    assert len(vectors) == 2


def test_embed_returns_correct_dimensions(provider):
    p, client = provider
    client.embeddings.create.return_value = _make_response(
        [[0.1, 0.2, 0.3, 0.4], [0.5, 0.6, 0.7, 0.8]]
    )

    vectors = p.embed(["hello", "world"])

    assert all(len(v) == 4 for v in vectors)


def test_embed_raises_on_empty_input(provider):
    p, _ = provider
    with pytest.raises(ValueError, match="empty"):
        p.embed([])


def test_dim_property_matches_config(provider):
    p, _ = provider
    assert p.dim == 4


def test_embed_with_usage_returns_result(provider):
    p, client = provider
    client.embeddings.create.return_value = _make_response(
        [[1.0, 0.0, 0.0, 0.0]], prompt_tokens=5
    )

    result = p.embed_with_usage(["single text"])

    assert len(result.vectors) == 1
    assert result.model == "text-embedding-3-small"
    assert result.prompt_tokens == 5


def test_embed_uses_configured_model(provider):
    p, client = provider
    client.embeddings.create.return_value = _make_response([[0.0, 0.0, 0.0, 0.0]])

    p.embed(["test"])

    call_kwargs = client.embeddings.create.call_args
    assert call_kwargs.kwargs["model"] == "text-embedding-3-small"


def test_embed_batches_large_input():
    """Provider splits large input into batches of batch_size."""
    with patch("rag.infra.embedding.openai_embedding.OpenAI") as MockClient:
        mock_client = MagicMock()
        MockClient.return_value = mock_client
        p = OpenAIEmbeddingProvider(model="text-embedding-3-small", dim=2, batch_size=2)
        p._client = mock_client

        # 5 texts, batch_size=2 → 3 API calls
        def side_effect(**kwargs):
            batch = kwargs["input"]
            return _make_response([[0.0, 0.0]] * len(batch))

        mock_client.embeddings.create.side_effect = side_effect

        vectors = p.embed(["a", "b", "c", "d", "e"])

        assert len(vectors) == 5
        assert mock_client.embeddings.create.call_count == 3


class TestConcurrentBatchEmbedding:
    """Batches are network-bound (one HTTP round-trip each), so they're sent
    concurrently via a thread pool — same reasoning as Vision page
    transcription (PdfVisionParser.parse_pages_parallel)."""

    def test_vectors_preserve_input_order_despite_concurrent_completion(self):
        # Each batch's fake "API call" sleeps a different amount so batches
        # complete out of submission order — the returned vectors must still
        # match the original text order, not completion order.
        import time

        with patch("rag.infra.embedding.openai_embedding.OpenAI") as MockClient:
            mock_client = MagicMock()
            MockClient.return_value = mock_client
            p = OpenAIEmbeddingProvider(
                model="text-embedding-3-small", dim=1, batch_size=1, max_workers=4
            )
            p._client = mock_client

            # 4 single-text batches; later-submitted batches finish first.
            delays = {"a": 0.06, "b": 0.04, "c": 0.02, "d": 0.0}

            def side_effect(**kwargs):
                text = kwargs["input"][0]
                time.sleep(delays[text])
                return _make_response([[ord(text)]])

            mock_client.embeddings.create.side_effect = side_effect

            vectors = p.embed(["a", "b", "c", "d"])

            assert vectors == [[ord("a")], [ord("b")], [ord("c")], [ord("d")]]

    def test_batches_run_concurrently_bounded_by_max_workers(self):
        # Track peak concurrently-active fake API calls to prove batches
        # actually overlap (not accidentally serialized) while respecting
        # the max_workers cap.
        import threading
        import time

        lock = threading.Lock()
        state = {"active": 0, "peak": 0}

        with patch("rag.infra.embedding.openai_embedding.OpenAI") as MockClient:
            mock_client = MagicMock()
            MockClient.return_value = mock_client
            p = OpenAIEmbeddingProvider(
                model="text-embedding-3-small", dim=1, batch_size=1, max_workers=3
            )
            p._client = mock_client

            def side_effect(**kwargs):
                with lock:
                    state["active"] += 1
                    state["peak"] = max(state["peak"], state["active"])
                time.sleep(0.05)
                with lock:
                    state["active"] -= 1
                return _make_response([[0.0]])

            mock_client.embeddings.create.side_effect = side_effect

            p.embed([str(i) for i in range(6)])

            assert state["peak"] > 1  # actually ran concurrently
            assert state["peak"] <= 3  # respected max_workers

    def test_total_tokens_aggregated_across_concurrent_batches(self):
        with patch("rag.infra.embedding.openai_embedding.OpenAI") as MockClient:
            mock_client = MagicMock()
            MockClient.return_value = mock_client
            p = OpenAIEmbeddingProvider(
                model="text-embedding-3-small", dim=1, batch_size=1, max_workers=4
            )
            p._client = mock_client

            def side_effect(**kwargs):
                return _make_response([[0.0]], prompt_tokens=7)

            mock_client.embeddings.create.side_effect = side_effect

            result = p.embed_with_usage(["a", "b", "c"])

            assert result.prompt_tokens == 21
