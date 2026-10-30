"""The public release keeps every training field and drops every score of the test turns."""
import json

from sessionroutex.arena import data
from sessionroutex.arena.release import release
from synth import MODELS, write_dataset


def rows(path):
    return [json.loads(x) for x in open(path)]


def test_release_drops_every_test_score(tmp_path):
    src = write_dataset(tmp_path / "data")
    stats = release(src, tmp_path / "public", tmp_path / "hidden")
    before, after = rows(src / "chat.jsonl"), rows(tmp_path / "public" / "chat.jsonl")
    assert len(before) == len(after) and stats["turns"]["test"] == sum(r["split"] == "test" for r in before)
    for b, a in zip(before, after):
        if b["split"] == "train":
            assert a == b                                        # training turns: every field
            continue
        text = json.dumps(a)
        for word in ("judge", "quality", "score", "labels"):
            assert word not in text, word
        assert all("text" not in r for r in a["responses"].values())
        assert {m: (r["n_prompt_tokens"], r["n_output_tokens"]) for m, r in a["responses"].items()} == \
            {m: (r["n_prompt_tokens"], r["n_output_tokens"]) for m, r in b["responses"].items()}
        for k in ("conversation_id", "turn", "history", "query", "original_answer", "split"):
            assert a[k] == b[k]
    assert (tmp_path / "public" / "embeddings" / "emb_chat_granite3b.npz").exists()
    assert not (tmp_path / "public" / "labels.jsonl").exists()
    hidden = rows(tmp_path / "hidden" / "labels.jsonl")
    assert len(hidden) == stats["turns"]["test"] and "judge_quality" in json.dumps(hidden)


def test_hidden_labels_join_back(tmp_path):
    src = write_dataset(tmp_path / "data")
    release(src, tmp_path / "public", tmp_path / "hidden")
    full, _ = data.load("chat", MODELS, data=src)
    joined, _ = data.load("chat", MODELS, data=tmp_path / "public", hidden=tmp_path / "hidden")
    public, _ = data.load("chat", MODELS, data=tmp_path / "public")
    assert joined == full
    assert {c.split for c in public} == {"train"} and len(public) < len(full)


def test_release_refuses_to_publish_inside_the_data_or_next_to_the_labels(tmp_path):
    import pytest
    src = write_dataset(tmp_path / "data")
    with pytest.raises(SystemExit):
        release(src, src / "public")
    with pytest.raises(SystemExit):
        release(src, tmp_path / "public", tmp_path / "public" / "hidden")
