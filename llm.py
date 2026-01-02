import requests
import json
import threading
from dotenv import load_dotenv
import os
from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    retry_if_not_exception_type,
)
from logger import get_logger

load_dotenv(override=True)
logger = get_logger("benchmarking")


class AuthenticationError(Exception):
    pass


class LLM:
    _models_cache = None
    _cache_lock = threading.Lock()

    def __init__(self, model_name: str = ""):
        self.api_key = os.getenv("OPENROUTER_API_KEY")
        self.base_url = "https://openrouter.ai/api/v1"
        self.headers = {
            "Authorization": f"Bearer {self.api_key}",
            "HTTP-Referer": "https://github.com/kumar-shivang/benchmarking",
            "X-Title": "Benchmarking Tool",
            "Content-Type": "application/json",
        }
        self.model_name = model_name
        if model_name:
            if model_name not in self.get_models():
                error_msg = f"Model {model_name} is not available."
                logger.error(error_msg)
                raise ValueError(error_msg)
            logger.debug(f"Initialized LLM with model: {model_name}")

    @retry(
        stop=stop_after_attempt(10),
        wait=wait_exponential(multiplier=1, min=4, max=60),
        retry=retry_if_not_exception_type(AuthenticationError),
        reraise=True,
    )
    def generate_response(self, messages: list, response_format: dict = None):
        payload = {
            "model": self.model_name,
            "messages": messages,
            "provider": {
                "sort": "throughput",
            },
        }
        try:
            logger.debug(f"Making API call to {self.model_name}")
            response = requests.post(
                f"{self.base_url}/chat/completions",
                headers=self.headers,
                data=json.dumps(payload),
                timeout=600,
            )

            if response.status_code == 401:
                logger.error(f"Authentication failed for model {self.model_name}")
                raise AuthenticationError("Invalid API Key or unauthorized access.")

            response.raise_for_status()
            data = response.json()

            content = data["choices"][0]["message"]["content"]
            usage = data.get("usage", {})

            usage_data = {
                "prompt_tokens": usage.get("prompt_tokens", 0),
                "completion_tokens": usage.get("completion_tokens", 0),
                "total_tokens": usage.get("total_tokens", 0),
                "cost": usage.get("cost", 0.0),
            }

            logger.debug(
                f"API call successful - Model: {self.model_name}, Cost: ${usage_data['cost']:.4f}, Tokens: {usage_data['total_tokens']}"
            )
            return content, usage_data

        except AuthenticationError:
            raise
        except requests.exceptions.Timeout as e:
            logger.warning(f"API call timeout for {self.model_name}: {str(e)}")
            raise
        except requests.exceptions.RequestException as e:
            logger.error(
                f"API request failed for {self.model_name}: {str(e)}", exc_info=True
            )
            try:
                logger.debug("Fetching available models from OpenRouter API")
                response = requests.get(
                    f"{self.base_url}/models", headers=self.headers, timeout=60
                )
                response.raise_for_status()
                data = response.json()
                with LLM._cache_lock:
                    LLM._models_cache = [model["id"] for model in data["data"]]
                logger.info(f"Successfully fetched {len(LLM._models_cache)} models")
            except Exception as e:
                logger.error(f"Failed to fetch models: {str(e)}", exc_info=True)
                raise
        return content, usage_data

    @retry(
        stop=stop_after_attempt(10),
        wait=wait_exponential(multiplier=1, min=4, max=60),
        retry=retry_if_not_exception_type(AuthenticationError),
        reraise=True,
    )
    def get_models(self):
        if LLM._models_cache is None:
            with LLM._cache_lock:
                # Double-check pattern
                if LLM._models_cache is None:
                    response = requests.get(
                        f"{self.base_url}/models", headers=self.headers, timeout=60
                    )
                    response.raise_for_status()
                    data = response.json()
                    LLM._models_cache = [model["id"] for model in data["data"]]
        return LLM._models_cache
