"""Dynamic track: run, judge, score and submit against fake in-process OpenAI-compatible endpoints."""
import json
import tarfile

import pytest

from fakes import PRICES, FakeOpenAI, Scripted  # noqa: F401
from sessionroutex.arena import dynamic
from sessionroutex.cli import main
from synth import write_dataset


@pytest.fixture
def setup(tmp_path):
    fake = FakeOpenAI(user_turns=2)
    root = write_dataset(tmp_path / "data", n_conv=3)
    cfg = {"router": "scripted-test", "serving": ["vllm"], "models": {"vllm": ["small", "big"]},
           "backends": {"vllm": {"base_url": fake.url, "prices": PRICES}}}
    (tmp_path / "pool.yaml").write_text(json.dumps(cfg))
    yield fake, root, tmp_path
    fake.close()


def test_run_judge_score_submit(setup):
    fake, root, tmp = setup
    log = tmp / "runs" / "dyn"
    main(["arena", "run", "--config", str(tmp / "pool.yaml"), "--tasks", str(root / "chat.jsonl"), "--out", str(log),
          "--lams", "0.1", "0.9", "--max-turns", "5", "--user-url", fake.url, "--user-model", "fake-user"])
    rows = [json.loads(x) for x in open(log / "log.jsonl")]
    # 3 tasks x 2 cost weights; the user ends each chat after the first message and 2 follow-ups
    assert len(rows) == 3 * 2 * 3 and {r["turn"] for r in rows} == {1, 2, 3}
    first = [r for r in rows if r["turn"] == 1]
    assert {r["query"] for r in first} == {json.loads(x)["query"] for x in open(root / "chat.jsonl")
                                           if json.loads(x)["turn"] == 1}
    assert rows[1]["query"].startswith("follow-up") and rows[1]["n_cached_tokens"] == 15
    assert all(r["cost"] > 0 and r["model"] in ("small", "big") for r in rows)
    user_reqs = [q for q in fake.requests if q["body"]["model"] == "fake-user"]
    assert user_reqs and user_reqs[0]["body"]["temperature"] == 0.0
    assert "Goal:" in user_reqs[0]["body"]["messages"][1]["content"]

    main(["arena", "run", "--config", str(tmp / "pool.yaml"), "--tasks", str(root / "chat.jsonl"), "--out", str(log),
          "--lams", "0.1", "0.9", "--user-url", fake.url, "--user-model", "fake-user"])
    assert len(open(log / "log.jsonl").readlines()) == len(rows)            # resumed: nothing played twice

    main(["arena", "judge", "--log", str(log), "--judge-url", fake.url, "--judge-model", "fake-judge"])
    judged = [json.loads(x) for x in open(log / "judged.jsonl")]
    assert len(judged) == len(rows) and {j["judge_quality"] for j in judged} == {0.75}
    jreq = [q for q in fake.requests if q["body"]["model"] == "fake-judge"]
    assert "Assistant reply to rate" in jreq[-1]["body"]["messages"][1]["content"]

    main(["arena", "score", "--log", str(log)])
    s = json.loads((log / "score.json").read_text())
    assert [p["lam"] for p in s["points"]] == [0.1, 0.9] and all(p["quality"] == 0.75 for p in s["points"])
    assert all(0 < p["cost"] <= 1 for p in s["points"]) and s["peak_quality"] == 0.75
    assert s["bill_usd"] == pytest.approx(sum(r["cost"] for r in rows))
    res = dynamic.score(str(log), reference=str(log))
    assert res["QNC_hull"] == min(p["cost"] for p in s["points"])

    main(["arena", "submit", "--log", str(log)])
    with tarfile.open(log / "submission.tar.gz") as t:
        assert sorted(t.getnames()) == ["judged.jsonl", "log.jsonl", "meta.json", "score.json"]
    meta = json.loads((log / "meta.json").read_text())
    assert meta["user_model"] == "fake-user" and meta["prices"]["big"] == PRICES["big"]


def test_defaults_and_score_parsing():
    assert dynamic.USER_MODEL == "DeepSeek-V4.1-Flash-1017" and "Nemotron-3.5-Super" in dynamic.JUDGE_MODEL
    assert len(dynamic.LAMS) == 5 and dynamic.LAMS[2] == 0.5
    assert dynamic.parse_score("Score: 8/9") == 8 and dynamic.parse_score("no idea") is None
