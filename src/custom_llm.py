import logging
import os
import time
from typing import List, Optional

import requests
from dotenv import load_dotenv
from langchain_core.language_models import LLM
from langchain_core.prompts import PromptTemplate

from src.utils import setup_logging

load_dotenv(override=True)
setup_logging()
logger = logging.getLogger(__name__)

config = os.environ


# import yaml
# with open("config.yaml", "r") as f:
#     config = yaml.safe_load(f)


class CustomLLM(LLM):
    """Custom LLM wrapper for LangChain using a REST API."""
    model: str
    endpoint_url: str = config.get("API_URL", "")
    headers: dict
    temperature: float = 0.7
    top_p: float = 1.0
    max_tokens: int = 2000
    stop: Optional[List[str]] = None

    def _call(self, prompt: str, stop: Optional[List[str]] = None) -> str:
        payload = {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": self.temperature,
            "top_p": self.top_p,
            "max_tokens": self.max_tokens,
        }
        logger.info(f'[INFO] Model : {self.model} | temp : {self.temperature} | top_p : {self.top_p} | max_token : {self.max_tokens}')
        if stop:
            payload["stop"] = stop

        max_retries = 5
        for attempt in range(max_retries):
            try:
                logger.info(f'[INFO] Header :  {self.headers}')
                logger.info(f'[INFO] Endpoint_url :  {self.endpoint_url}')
                response = requests.post(self.endpoint_url, headers=self.headers, json=payload)
                response.raise_for_status()
                data = response.json()
                return data['choices'][0]['message']['content']
            except requests.exceptions.HTTPError as http_err:
                response = getattr(http_err, "response", None)
                if response is not None and response.status_code == 429:
                    wait_time = 2 ** attempt
                    print(f"[Retry {attempt + 1}/{max_retries}] Rate limited. Retrying in {wait_time}s...")
                    logger.info(f"[INFO] [Retry {attempt + 1}/{max_retries}] Rate limited. Retrying in {wait_time}s...")
                    time.sleep(wait_time)
                else:
                    logger.error(f'[ERROR] HTTP error occurred: {http_err}')
                    print(f"HTTP error occurred: {http_err}")
                    raise
            except Exception as e:
                logger.error(f"[ERROR] Unexpected error occurred: {e}")
                print(f"Unexpected error occurred: {e}")
                raise
        logger.error("[ERROR] Exceeded maximum retry attempts for LLM call.")
        raise Exception("Exceeded maximum retry attempts for LLM call.")

    @property
    def _llm_type(self) -> str:
        return "custom-llm"


# Initialize custom LLM (default instance; chat uses get_llm() to pass sidebar temperature)
llm = CustomLLM(
    model=config.get("LLM_MODEL", "gpt-5-mini"),
    endpoint_url=config.get("API_URL", ""),
    headers={"X-API-KEY": config.get("API_KEY", "")},
    temperature=0.5,
    top_p=0.9,
    max_tokens=1000
)

# Prompt Template (example; RAG chat uses its own prompt in chat.py)
prompt = PromptTemplate.from_template("You are helpful AI. Reply to: {text}")


def get_llm(
    temperature: float = 0.5,
    top_p: float = 0.9,
    max_tokens: int = 1000,
    model: Optional[str] = None,
    endpoint_url: Optional[str] = None,
    headers: Optional[dict] = None,
) -> CustomLLM:
    """Returns a CustomLLM instance (used by chat to pass sidebar temperature)."""
    return CustomLLM(
        model=model or config.get("LLM_MODEL", "gpt-5-mini"),
        endpoint_url=endpoint_url or config.get("API_URL", ""),
        headers=headers or {"X-API-KEY": config.get("API_KEY", "")},
        temperature=temperature,
        top_p=top_p,
        max_tokens=max_tokens,
    )


# === Run with example input ===
if __name__ == "__main__":
    user_input = prompt.format(text="What is Quantum Computing?")
    print('>>> ', user_input)
    response = llm.invoke(user_input)
    print("LLM Response:", response)
