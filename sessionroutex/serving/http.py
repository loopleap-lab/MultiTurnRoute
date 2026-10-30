"""A small JSON-over-HTTP helper (standard library only) for OpenAI-compatible endpoints."""
import json
import os
import urllib.error
import urllib.request


def post_json(url: str, payload: dict, headers: dict | None = None, timeout: float = 120) -> dict:
    req = urllib.request.Request(url, data=json.dumps(payload).encode(), method="POST",
                                 headers={"Content-Type": "application/json", **(headers or {})})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return json.loads(r.read().decode())
    except urllib.error.HTTPError as e:
        raise RuntimeError(f"{url}: HTTP {e.code}: {e.read().decode(errors='replace')[:500]}") from None


def auth_headers(api_key_env: str | None) -> dict:
    """Authorization from an environment variable; the key itself is never stored or logged."""
    key = os.environ.get(api_key_env) if api_key_env else None
    return {"Authorization": f"Bearer {key}"} if key else {}


def chat_completion(base_url: str, model: str, messages: list, api_key_env: str | None = None, timeout: float = 120,
                    **extra) -> dict:
    """POST {base_url}/chat/completions and return the response JSON."""
    return post_json(base_url.rstrip("/") + "/chat/completions", {"model": model, "messages": messages, **extra},
                     auth_headers(api_key_env), timeout)
