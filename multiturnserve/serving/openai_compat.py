"""Any OpenAI-compatible chat completions server. The cached part of the prompt comes from
usage.prompt_tokens_details.cached_tokens (0 when the server does not report it)."""
from ..types import Usage
from .base import Serving
from .http import chat_completion


class OpenAICompatible(Serving):
    base_url: str = ""
    api_key_env: str | None = None
    extra: dict = {}

    def __init__(self, models: list | None = None, prices: dict | None = None, base_url: str | None = None,
                 api_key_env: str | None = None, model_ids: dict | None = None, timeout: float = 120,
                 extra: dict | None = None):
        super().__init__(models, prices)
        self.base_url = base_url or self.base_url
        self.api_key_env = api_key_env if api_key_env is not None else self.api_key_env
        self.model_ids = dict(model_ids or {})     # our model name -> the server's model id
        self.timeout = timeout
        self.extra = {**self.extra, **(extra or {})}

    def run(self, model: str, messages: list):
        r = chat_completion(self.base_url, self.model_ids.get(model, model), messages, self.api_key_env, self.timeout,
                            **self.extra)
        return r["choices"][0]["message"]["content"] or "", Usage.of(r["usage"])
