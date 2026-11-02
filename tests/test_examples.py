"""The website's files in examples/ import and run: my_router.py and my_backend.py behind a real gateway (uvicorn)
with the echo backend, app.py talking to it, and my_router.py in the static track's replay."""
import json
import runpy
import socket
import threading
import time
from pathlib import Path

import pytest

import multiturnserve as mts
from synth import write_dataset

EX = Path(__file__).resolve().parents[1] / "examples"


def free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


@pytest.fixture
def gateway(tmp_path):
    uvicorn = pytest.importorskip("uvicorn")
    from multiturnserve.gateway import create_app
    rt = mts.Runtime.from_config(str(EX / "config.yaml"), log=str(tmp_path / "logs"))
    port = free_port()
    server = uvicorn.Server(uvicorn.Config(create_app(rt), host="127.0.0.1", port=port, log_level="warning"))
    th = threading.Thread(target=server.run, daemon=True)
    th.start()
    for _ in range(100):
        if server.started:
            break
        time.sleep(0.05)
    yield f"http://127.0.0.1:{port}", rt, tmp_path
    server.should_exit = True
    th.join(5)


def test_app_routes_every_turn_through_the_gateway(gateway, monkeypatch, capsys):
    url, rt, tmp = gateway
    import multiturnserve.client as client
    real = client.gateway_url
    monkeypatch.setattr(client, "gateway_url", lambda g: url if g == ":8080" else real(g))
    runpy.run_path(str(EX / "app.py"), run_name="__main__")
    lines = capsys.readouterr().out.strip().splitlines()
    assert [ln.split()[0] for ln in lines] == ["1", "2", "3", "4"]
    assert all(ln.split()[1] in rt.models and float(ln.split()[2]) > 0 for ln in lines)
    st = rt.sessions["session-1"].summary()
    assert st["turns"] == 4 and len(open(tmp / "logs" / "turns.jsonl").readlines()) == 4


def test_my_router_and_my_backend_register():
    from multiturnserve.registry import get_router, get_serving
    R = get_router(str(EX / "my_router.py"))
    B = get_serving(str(EX / "my_backend.py"))
    assert R.name == "my-router" and B.name == "my-backend" and B.models == ["gemma-26b-a4b", "gemma-31b"]
    text, usage = B().run("gemma-31b", [{"role": "user", "content": "hello"}])
    assert text == "echo: hello" and mts.Usage.of(usage).prompt_tokens == 2
    assert B().prices("gemma-31b") == mts.pricing.PRICES["gemma-31b"]


def test_my_router_in_the_static_track(tmp_path):
    from multiturnserve.cli import main
    root = write_dataset(tmp_path / "data")
    main(["arena", "replay", "--router", str(EX / "my_router.py"), "--data", str(root), "--models", "gemma-e4b",
          "gemma-31b", "mistral-medium-128b", "--dev", "--out", str(tmp_path), "--name", "mine"])
    res = json.loads((tmp_path / "mine" / "results.json").read_text())
    assert set(res["policies"]) == {"my-router", "oracle (true quality and cost)"}
    assert 0 < res["policies"]["my-router"]["AUDC"] <= 1 and len(res["policies"]["my-router"]["curve"]) == 41
