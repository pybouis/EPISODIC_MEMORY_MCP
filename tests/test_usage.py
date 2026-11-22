"""Mesure des tokens : taille estimée et consommation lue dans un journal Claude Code factice."""

import json
from dataclasses import replace

import pytest

from episodic_memory.config import load_settings
from episodic_memory.store import EpisodicStore
from episodic_memory.usage import estimate_tokens, measure, normalize_reported, transcript_usage

from test_store import FakeEmbedder

Q1 = "Donne moi le P&L 2023 par division"
Q2 = "Calcule le cash par Unit"


def _user(ts, text):
    return {"type": "user", "timestamp": ts, "message": {"role": "user", "content": text}}


def _result(ts):
    return {"type": "user", "timestamp": ts, "message": {"role": "user", "content": [{"type": "tool_result"}]}}


def _assistant(ts, msg_id, tool=None, args=None, inp=10, out=5, cache_read=100):
    content = [{"type": "tool_use", "name": f"mcp__episodic-memory__{tool}", "input": args or {}}] if tool else [
        {"type": "text", "text": "..."}
    ]
    return {
        "type": "assistant",
        "timestamp": ts,
        "message": {
            "id": msg_id,
            "model": "claude-test",
            "content": content,
            "usage": {"input_tokens": inp, "output_tokens": out, "cache_read_input_tokens": cache_read,
                      "cache_creation_input_tokens": 0},
        },
    }


@pytest.fixture
def transcript(tmp_path):
    project = tmp_path / "projects" / "mon-projet"
    project.mkdir(parents=True)
    path = project / "session-1.jsonl"
    entries = [
        # Première question, hors fenêtre.
        _user("2026-01-01T10:00:00Z", Q2),
        _assistant("2026-01-01T10:00:01Z", "m0", "find_similar_question", {"question": Q2}, inp=1000, out=1000),
        _result("2026-01-01T10:00:02Z"),
        # Deuxième question : c'est elle qu'on stocke.
        _user("2026-01-01T10:01:00Z", Q1),
        _assistant("2026-01-01T10:01:01Z", "m1", "find_similar_question", {"question": Q1}),
        _result("2026-01-01T10:01:02Z"),
        _assistant("2026-01-01T10:01:03Z", "m2"),
        _assistant("2026-01-01T10:01:03Z", "m2"),  # même réponse API sur deux lignes : comptée une fois
        _user("2026-01-01T10:02:00Z", "oui enregistre"),
        _assistant("2026-01-01T10:02:01Z", "m3", "store_validated_episode", {"question": Q1}, out=50),
    ]
    path.write_text("\n".join(json.dumps(e) for e in entries) + "\n", encoding="utf-8")
    sub = project / "session-1" / "subagents"
    sub.mkdir(parents=True)
    (sub / "agent-a.jsonl").write_text(json.dumps(_assistant("2026-01-01T10:01:30Z", "s1", out=7)) + "\n", encoding="utf-8")
    return path


def test_estimate_tokens():
    assert estimate_tokens("") == 0
    assert estimate_tokens("abcd" * 10) == 10


def test_transcript_usage_window_dedup_and_subagents(transcript):
    usage = transcript_usage(transcript, Q1)
    assert usage["api_calls"] == 4  # m1, m2 (dédoublonné), m3, sous-agent
    assert usage["subagent_calls"] == 1
    assert usage["output_tokens"] == 5 + 5 + 50 + 7
    assert usage["input_tokens"] == 40
    assert usage["total_tokens"] == usage["input_tokens"] + usage["output_tokens"] + 400
    assert usage["window"] == "question" and usage["models"] == ["claude-test"]


def test_transcript_usage_unknown_question(transcript):
    assert transcript_usage(transcript, "autre question") is None


def test_measure_finds_recent_transcript(transcript):
    usage = measure(transcript.parents[1], Q1, wait_seconds=0)
    assert usage and usage["session_id"] == "session-1"
    assert measure(transcript.parents[1], "jamais stockée", wait_seconds=0) is None


def test_normalize_reported():
    u = normalize_reported({"input_tokens": 3, "output_tokens": 4, "model": "m"})
    assert u["total_tokens"] == 7 and u["source"] == "reported" and u["models"] == ["m"]


def test_store_keeps_usage_and_totals(tmp_path):
    settings = replace(
        load_settings(), db_dir=tmp_path / "chroma", export_dir=tmp_path / "exports",
        artifacts_dir=tmp_path / "artifacts", glossary_path=tmp_path / "g.json", user="t",
    )
    store = EpisodicStore(settings, embedder=FakeEmbedder())
    usage = normalize_reported({"input_tokens": 100, "output_tokens": 20})
    ep = store.add(Q1, "x" * 400, reasoning="y" * 40, usage=usage)
    assert ep["size_tokens"]["answer"] == 100 and ep["size_tokens"]["reasoning"] == 10
    assert ep["usage"]["total_tokens"] == 120
    # Remplacement sans nouvelle mesure : la mesure est conservée.
    ep2 = store.add(Q1, "z", replace_episode_id=ep["id"])
    assert ep2["usage"]["total_tokens"] == 120
    store.add(Q2, "réponse")
    totals = store.token_totals()
    assert totals["episodes"] == 2 and totals["measured_episodes"] == 1 and totals["consumed_tokens"] == 120
    assert {e["tokens"]["consumed"] for e in store.list()} == {120, None}
    # Export / import : la mesure suit.
    path = store.export_json()
    store.delete(ep["id"])
    store.import_json(path)
    assert store.get(ep["id"])["usage"]["total_tokens"] == 120


def test_post_tool_use_hook_asks_to_record_usage(transcript):
    from episodic_memory.hook import usage_context

    response = [{"type": "text", "text": json.dumps({"status": "stored", "episode": {"id": "ep_abc", "usage": None}})}]
    payload = {
        "hook_event_name": "PostToolUse",
        "tool_name": "mcp__episodic-memory__store_validated_episode",
        "tool_input": {"question": Q1},
        "tool_response": response,
        "transcript_path": str(transcript),
    }
    context = usage_context(payload)
    assert "set_episode_usage" in context and 'episode_id="ep_abc"' in context and '"output_tokens": 67' in context
    # Déjà mesuré par le serveur, échec du stockage ou autre outil : rien.
    measured = [{"type": "text", "text": json.dumps({"status": "stored", "episode": {"id": "ep_abc", "usage": {"x": 1}}})}]
    assert usage_context(payload | {"tool_response": measured}) is None
    failed = [{"type": "text", "text": json.dumps({"status": "duplicate_found"})}]
    assert usage_context(payload | {"tool_response": failed}) is None
    assert usage_context(payload | {"tool_name": "mcp__episodic-memory__list_episodes"}) is None
