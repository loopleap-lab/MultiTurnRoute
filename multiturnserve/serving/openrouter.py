"""OpenRouter (https://openrouter.ai). The API key comes from OPENROUTER_API_KEY."""
from ..registry import register_serving
from .openai_compat import OpenAICompatible


@register_serving
class OpenRouter(OpenAICompatible):
    name = "openrouter"
    base_url = "https://openrouter.ai/api/v1"
    api_key_env = "OPENROUTER_API_KEY"
    extra = {"usage": {"include": True}}
