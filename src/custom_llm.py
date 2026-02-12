"""Custom LLM that calls your REST API (OpenAI-compatible). No LangChain dependency."""
import logging
import os
import time
from typing import List, Optional

import requests
from dotenv import load_dotenv

from src.utils import setup_logging

load_dotenv(override=True)
setup_logging()
logger = logging.getLogger(__name__)
config = os.environ


class CustomLLM:
    def __init__(
        self,
        model: str,
        endpoint_url: str = "",
        headers: Optional[dict] = None,
        temperature: float = 0.7,
        top_p: float = 1.0,
        max_tokens: int = 2000,
        stop: Optional[List[str]] = None,
    ):
        self.model = model
        self.endpoint_url = (endpoint_url or config.get("API_URL", "")).strip()
        self.headers = headers or {"X-API-KEY": config.get("API_KEY", "")}
        self.temperature = temperature
        self.top_p = top_p
        self.max_tokens = max_tokens
        self.stop = stop

    def _call(self, prompt: str, stop: Optional[List[str]] = None) -> str:
        if not self.endpoint_url or not self.endpoint_url.startswith(("http://", "https://")):
            raise ValueError(
                "API_URL is not set or invalid. Copy .env.example to .env and set API_URL "
                "(e.g. https://your-api.com/v1/chat/completions)."
            )
        payload = {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": self.temperature,
            "top_p": self.top_p,
            "max_tokens": self.max_tokens,
        }
        if stop:
            payload["stop"] = stop
        logger.info("Model=%s temp=%s top_p=%s max_tokens=%s", self.model, self.temperature, self.top_p, self.max_tokens)
        max_retries = 5
        response = None
        for attempt in range(max_retries):
            try:
                response = requests.post(self.endpoint_url, headers=self.headers, json=payload, timeout=120)
                response.raise_for_status()
                return response.json()["choices"][0]["message"]["content"]
            except requests.exceptions.HTTPError as http_err:
                response = getattr(http_err, "response", None)
                if response is not None and response.status_code == 429:
                    wait_time = 2 ** attempt
                    logger.warning("Rate limited, retry %s/%s in %ss", attempt + 1, max_retries, wait_time)
                    time.sleep(wait_time)
                else:
                    logger.error("HTTP error: %s", http_err)
                    raise
            except Exception as e:
                logger.error("LLM request failed: %s", e)
                raise
        raise RuntimeError("Exceeded maximum retry attempts for LLM call.")

    def invoke(self, prompt: str, stop: Optional[List[str]] = None) -> str:
        return self._call(prompt, stop=stop or self.stop)


llm = CustomLLM(
    model=config.get("LLM_MODEL", "gpt-5-mini"),
    endpoint_url=config.get("API_URL", ""),
    headers={"X-API-KEY": config.get("API_KEY", "")},
    temperature=0.5,
    top_p=0.9,
    max_tokens=1000,
)


def get_llm(
    temperature: float = 0.5,
    top_p: float = 0.9,
    max_tokens: int = 1000,
    model: Optional[str] = None,
    endpoint_url: Optional[str] = None,
    headers: Optional[dict] = None,
) -> CustomLLM:
    return CustomLLM(
        model=model or config.get("LLM_MODEL", "gpt-5-mini"),
        endpoint_url=endpoint_url or config.get("API_URL", ""),
        headers=headers or {"X-API-KEY": config.get("API_KEY", "")},
        temperature=temperature,
        top_p=top_p,
        max_tokens=max_tokens,
    )


if __name__ == "__main__":
    print(llm.invoke("You are helpful AI. Reply to: What is 2+2?"))
