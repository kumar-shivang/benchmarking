import requests
import json
from dotenv import load_dotenv
import os
from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    retry_if_not_exception_type,
)

load_dotenv(override=True)


class AuthenticationError(Exception):
    pass


class LLM:
    _models_cache = None

    def __init__(self, model_name: str = ""):
        self.api_key = os.getenv("OPENROUTER_API_KEY")
        self.base_url = "https://openrouter.ai/api/v1"
        self.headers = {
            "Authorization": f"Bearer {self.api_key}",
            "HTTP-Referer": "https://github.com/shivangkumar/benchmarking",
            "X-Title": "Benchmarking Tool",
            "Content-Type": "application/json",
        }
        self.model_name = model_name
        if model_name:
            if model_name not in self.get_models():
                raise ValueError(f"Model {model_name} is not available.")

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=4, max=10),
        retry=retry_if_not_exception_type(AuthenticationError),
        reraise=True,
    )
    def generate_response(self, messages: list):
        payload = {
            "model": self.model_name,
            "messages": messages,
        }
        response = requests.post(
            f"{self.base_url}/chat/completions",
            headers=self.headers,
            data=json.dumps(payload),
        )

        if response.status_code == 401:
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

        return content, usage_data

    def get_models(self):
        if LLM._models_cache is None:
            response = requests.get(f"{self.base_url}/models", headers=self.headers)
            response.raise_for_status()
            data = response.json()
            LLM._models_cache = [model["id"] for model in data["data"]]
        return LLM._models_cache
