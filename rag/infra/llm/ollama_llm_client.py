"""Ollama LLM client implementing the framework's BaseLLMClient."""

import json
import logging
import time
from typing import Any, Optional
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from rag.core.interfaces.llm_client import BaseLLMClient, LLMResponse

logger = logging.getLogger(__name__)


class OllamaLLMClient(BaseLLMClient):
    """Local LLM client backed by the Ollama HTTP API."""

    def __init__(
        self,
        model: str = "llama3.2",
        base_url: str = "http://localhost:11434",
        max_tokens: int = 1024,
        temperature: float = 0.2,
        system_prompt: Optional[str] = None,
        timeout: int = 180,
        **kwargs: Any,
    ) -> None:
        self._model = model
        self._base_url = base_url.rstrip("/")
        self._max_tokens = max_tokens
        self._temperature = temperature
        self._system_prompt = system_prompt
        self._timeout = timeout

    @property
    def model(self) -> str:
        return self._model

    def generate(self, prompt: str, **kwargs: Any) -> LLMResponse:
        messages = []
        if self._system_prompt:
            messages.append({"role": "system", "content": self._system_prompt})
        messages.append({"role": "user", "content": prompt})

        options = {
            "num_predict": kwargs.pop("max_tokens", self._max_tokens),
            "temperature": kwargs.pop("temperature", self._temperature),
        }
        options.update(kwargs.pop("options", {}))

        payload = {
            "model": self._model,
            "messages": messages,
            "stream": False,
            "options": options,
            **kwargs,
        }

        request = Request(
            f"{self._base_url}/api/chat",
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )

        start = time.perf_counter()
        try:
            with urlopen(request, timeout=self._timeout) as response:
                result = json.loads(response.read().decode("utf-8"))
        except HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")
            raise RuntimeError(f"Ollama HTTP error {exc.code}: {detail}") from exc
        except URLError as exc:
            raise RuntimeError(
                f"Cannot connect to Ollama at {self._base_url}. "
                "Make sure Ollama is running."
            ) from exc

        latency_ms = (time.perf_counter() - start) * 1000
        usage_prompt = int(result.get("prompt_eval_count") or 0)
        usage_completion = int(result.get("eval_count") or 0)

        return LLMResponse(
            text=result.get("message", {}).get("content", ""),
            model=result.get("model", self._model),
            prompt_tokens=usage_prompt,
            completion_tokens=usage_completion,
            total_tokens=usage_prompt + usage_completion,
            latency_ms=latency_ms,
        )

    def generate_structured(
        self,
        prompt: str,
        schema: dict[str, Any],
        **kwargs: Any,
    ) -> Optional[dict[str, Any]]:
        response = self.generate(prompt, format=schema, **kwargs)
        try:
            parsed = json.loads(response.text)
            return parsed if isinstance(parsed, dict) else None
        except (json.JSONDecodeError, ValueError):
            logger.warning("Ollama returned invalid structured JSON.")
            return None

    def count_tokens(self, text: str) -> int:
        try:
            import tiktoken
            return len(tiktoken.get_encoding("cl100k_base").encode(text))
        except Exception:
            return max(1, len(text) // 4)
