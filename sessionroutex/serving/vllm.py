"""vLLM's OpenAI-compatible server (vllm serve MODEL). Start it with --enable-prompt-tokens-details so the usage
reports cached tokens; an API key, if the server wants one, comes from VLLM_API_KEY."""
from ..registry import register_serving
from .openai_compat import OpenAICompatible


@register_serving
class VLLM(OpenAICompatible):
    name = "vllm"
    base_url = "http://localhost:8000/v1"
    api_key_env = "VLLM_API_KEY"
