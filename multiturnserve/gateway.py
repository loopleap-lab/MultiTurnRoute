"""`multiturnserve serve`: an HTTP gateway with an OpenAI-compatible /v1/chat/completions. Each request names its chat
(field "chat_id", header X-Chat-Id, or the OpenAI "user" field); the gateway keeps one session per chat, routes the
turn, runs the chosen model and updates the chat's state from the reported usage. Needs the [serve] extra.

  POST /v1/chat/completions   {chat_id, messages}  -> chat.completion (+ "multiturnserve": turn, model, cost)
  POST /v1/route              {chat_id, messages}  -> {model, quality, cost, turn}   (no state change)
  POST /v1/observe            {chat_id, model, text, usage[, cost]}  -> the chat's state
  GET  /v1/sessions/{chat_id}                      -> the chat's state
  GET  /v1/models, /health
"""
from __future__ import annotations

import time
import uuid

from .types import Outcome, Usage


SERVE_EXTRA = 'the gateway needs the [serve] extra: pip install -e ".[serve]"'


def create_app(runtime):
    try:
        from fastapi import Body, FastAPI, Header, HTTPException
    except ImportError:
        raise ImportError(SERVE_EXTRA) from None

    app = FastAPI(title="MultiTurnServe gateway")

    def chat_of(body: dict, header: str | None) -> str:
        cid = body.get("chat_id") or header or body.get("user")
        if not cid:
            raise HTTPException(400, "name the chat: a chat_id field, an X-Chat-Id header, or the user field")
        return str(cid)

    def messages_of(body: dict) -> list:
        msgs = body.get("messages")
        if not msgs or not isinstance(msgs, list):
            raise HTTPException(400, "messages: a non-empty list of {role, content}")
        return msgs

    @app.get("/health")
    def health():
        return {"router": runtime.router.name, "models": runtime.models}

    @app.get("/v1/models")
    def models():
        return {"object": "list", "data": [{"id": m, "object": "model", "owned_by": runtime.serving[m].name}
                                           for m in runtime.models]}

    @app.post("/v1/chat/completions")
    def completions(body: dict = Body(...), x_chat_id: str | None = Header(None)):
        cid, msgs = chat_of(body, x_chat_id), messages_of(body)
        s = runtime.session(cid)
        with s.lock:
            d = s.route(msgs)
            try:
                o = s.run(d, msgs)
            except Exception as e:  # the backend failed; the chat's state is unchanged
                s._pending = None
                raise HTTPException(502, f"{d.model} on {runtime.serving[d.model].name}: {e}") from None
            s.observe(o)
        u = o.usage
        return {"id": f"chatcmpl-{uuid.uuid4().hex[:24]}", "object": "chat.completion", "created": int(time.time()),
                "model": o.model,
                "choices": [{"index": 0, "message": {"role": "assistant", "content": o.text}, "finish_reason": "stop"}],
                "usage": {"prompt_tokens": u.prompt_tokens, "completion_tokens": u.output_tokens,
                          "total_tokens": u.prompt_tokens + u.output_tokens,
                          "prompt_tokens_details": {"cached_tokens": u.cached_tokens}},
                "multiturnserve": {"chat_id": cid, "turn": d.t + 1, "model": o.model, "cost": o.cost,
                                  "decision": {"quality": d.quality, "cost": d.cost}}}

    @app.post("/v1/route")
    def route(body: dict = Body(...), x_chat_id: str | None = Header(None)):
        s = runtime.session(chat_of(body, x_chat_id))
        with s.lock:
            d = s.route(messages_of(body))
        return {"model": d.model, "quality": d.quality, "cost": d.cost, "turn": d.t + 1}

    @app.post("/v1/observe")
    def observe(body: dict = Body(...), x_chat_id: str | None = Header(None)):
        s = runtime.session(chat_of(body, x_chat_id))
        if body.get("model") not in runtime.serving:
            raise HTTPException(400, f"model: one of {runtime.models}")
        o = Outcome(body["model"], str(body.get("text", "")), Usage.of(body.get("usage") or {"prompt_tokens": 0}),
                    body.get("cost"), body.get("query"))
        with s.lock:
            s.observe(o)
        return s.summary()

    @app.get("/v1/sessions/{chat_id}")
    def session(chat_id: str):
        if chat_id not in runtime.sessions:
            raise HTTPException(404, f"no chat {chat_id!r}")
        return runtime.sessions[chat_id].summary()

    return app


def serve(runtime, host: str = "127.0.0.1", port: int = 8080) -> None:
    try:
        import uvicorn
    except ImportError:
        raise ImportError(SERVE_EXTRA) from None
    uvicorn.run(create_app(runtime), host=host, port=port, log_level="info")
