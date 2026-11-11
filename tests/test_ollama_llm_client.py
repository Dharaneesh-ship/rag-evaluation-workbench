import json
from unittest.mock import patch
from urllib.error import URLError

import pytest

from rag.infra.llm.ollama_llm_client import OllamaLLMClient


@patch("rag.infra.llm.ollama_llm_client.urlopen")
def test_generate_returns_response_and_usage(mock_urlopen):
    mock_urlopen.return_value.__enter__.return_value.read.return_value = (
        json.dumps({
            "model": "llama3.2",
            "message": {"content": "Hello from Ollama"},
            "prompt_eval_count": 12,
            "eval_count": 8,
        }).encode()
    )

    client = OllamaLLMClient(model="llama3.2")
    result = client.generate("Say hello")

    assert result.text == "Hello from Ollama"
    assert result.model == "llama3.2"
    assert result.prompt_tokens == 12
    assert result.completion_tokens == 8
    assert result.total_tokens == 20
    assert result.latency_ms >= 0


@patch("rag.infra.llm.ollama_llm_client.urlopen")
def test_generate_includes_system_prompt(mock_urlopen):
    mock_urlopen.return_value.__enter__.return_value.read.return_value = (
        json.dumps({
            "model": "llama3.2",
            "message": {"content": "OK"},
        }).encode()
    )

    client = OllamaLLMClient(system_prompt="Be concise")
    client.generate("Test")

    request = mock_urlopen.call_args.args[0]
    payload = json.loads(request.data)

    assert payload["messages"][0] == {
        "role": "system",
        "content": "Be concise",
    }


@patch("rag.infra.llm.ollama_llm_client.urlopen")
def test_generate_structured_returns_json(mock_urlopen):
    mock_urlopen.return_value.__enter__.return_value.read.return_value = (
        json.dumps({
            "model": "llama3.2",
            "message": {"content": '{"score": 9}'},
        }).encode()
    )

    client = OllamaLLMClient()
    result = client.generate_structured(
        "Return a score",
        {"type": "object", "properties": {"score": {"type": "integer"}}},
    )

    assert result == {"score": 9}


@patch("rag.infra.llm.ollama_llm_client.urlopen", side_effect=URLError("offline"))
def test_connection_error_is_readable(mock_urlopen):
    client = OllamaLLMClient()

    with pytest.raises(RuntimeError, match="Cannot connect to Ollama"):
        client.generate("Test")


def test_count_tokens_returns_positive_integer():
    client = OllamaLLMClient()
    assert client.count_tokens("RAG evaluation workbench") > 0
