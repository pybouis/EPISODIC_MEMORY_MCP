"""Tests de la base avec un embedding factice (sac de mots haché) : pas de téléchargement."""

import hashlib
import math
import re
from dataclasses import replace

import pytest

from episodic_memory.config import load_settings
from episodic_memory.store import DuplicateEpisode, EpisodicStore, ReasoningTooLong


class FakeEmbedder:
    name = "fake-bow"

    def embed(self, texts):
        out = []
        for text in texts:
            vec = [0.0] * 64
            for word in re.findall(r"\w+", text.lower()):
                vec[int(hashlib.md5(word.encode()).hexdigest(), 16) % 64] += 1.0
            norm = math.sqrt(sum(v * v for v in vec)) or 1.0
            out.append([v / norm for v in vec])
        return out


@pytest.fixture
def store(tmp_path):
    settings = replace(
        load_settings(),
        db_dir=tmp_path / "chroma",
        export_dir=tmp_path / "exports",
        artifacts_dir=tmp_path / "artifacts",
        user="testeur",
        threshold=0.6,
        keyword_bonus=0.2,
        keyword_floor=0.2,
        glossary_path=tmp_path / "glossary.json",
    )
    return EpisodicStore(settings, embedder=FakeEmbedder())


Q = "Comment créer un cube dans TM1 avec TurboIntegrator ?"
A = "Utiliser CubeCreate dans l'onglet Prolog."


def test_empty_base_has_no_match(store):
    assert store.search("n'importe quoi") == []


def test_store_records_who_and_when(store):
    ep = store.add(Q, A)
    assert ep["validated_by"] == "testeur"
    assert ep["validated_at"] and ep["created_at"] == ep["validated_at"]
    assert ep["revision"] == 1


def test_search_finds_close_question_without_answer(store):
    ep = store.add(Q, A)
    store.add("Quelle est la capitale de la Belgique ?", "Bruxelles.")
    matches = store.search("Comment créer un cube TM1 avec TurboIntegrator")
    assert [m["id"] for m in matches] == [ep["id"]]
    assert "answer" not in matches[0]
    assert matches[0]["similarity"] >= 0.6


def test_get_returns_answer_and_counts_use(store):
    ep = store.add(Q, A)
    got = store.get(ep["id"], mark_used=True)
    assert got["answer"] == A
    assert got["use_count"] == 1 and got["last_used_at"]


def test_duplicate_is_detected_then_replaced(store):
    ep = store.add(Q, A)
    with pytest.raises(DuplicateEpisode) as exc:
        store.add(Q, "Autre réponse")
    assert exc.value.existing["id"] == ep["id"]

    new = store.add(Q, "Réponse améliorée", validated_by="alice", replace_episode_id=ep["id"])
    assert new["id"] == ep["id"] and new["revision"] == 2
    assert new["validated_by"] == "alice" and new["created_by"] == "testeur"
    assert store.count() == 1
    assert store.get(ep["id"])["answer"] == "Réponse améliorée"


def test_export_import_roundtrip(store, tmp_path):
    ep = store.add(Q, A, reasoning=R, tags=["tm1"])
    path = store.export_json()
    assert store.delete(ep["id"]) and store.count() == 0
    assert store.import_json(path) == 1
    got = store.get(ep["id"])
    assert got["answer"] == A and got["tags"] == ["tm1"] and got["validated_by"] == "testeur"
    assert got["reasoning"] == R


R = "**Sources** : cube Ventes.\n**Démarche** : 1. CubeExists 2. CubeCreate.\n**Pièges** : ordre des dimensions."


def test_reasoning_is_stored_and_flagged_in_search(store):
    ep = store.add(Q, A, reasoning=R)
    assert store.get(ep["id"])["reasoning"] == R
    match = store.search(Q)[0]
    assert match["has_reasoning"] is True and "reasoning" not in match


def test_reasoning_over_limit_is_refused(store):
    with pytest.raises(ReasoningTooLong) as exc:
        store.add(Q, A, reasoning="x" * (store.settings.reasoning_max_chars + 1))
    assert exc.value.max_chars == store.settings.reasoning_max_chars
    assert store.count() == 0


def test_artifacts_are_archived_and_survive_the_original(store, tmp_path):
    src = tmp_path / "rapport.xlsx"
    src.write_bytes(b"contenu excel")
    ep = store.add(Q, A, reasoning=R, artifacts=[{"path": str(src), "description": "Rapport des ventes"}])
    src.unlink()  # l'original disparaît, la copie d'archive reste

    art = store.get(ep["id"])["artifacts"][0]
    assert art["name"] == "rapport.xlsx" and art["description"] == "Rapport des ventes"
    assert art["original_path"] == str(src)
    assert art["path"] and open(art["path"], "rb").read() == b"contenu excel"
    assert store.search(Q)[0]["artifact_names"] == ["rapport.xlsx"]

    archive_dir = store.settings.artifacts_dir / ep["id"]
    assert store.delete(ep["id"]) and not archive_dir.exists()


def test_missing_artifact_is_refused(store, tmp_path):
    with pytest.raises(ValueError, match="introuvable"):
        store.add(Q, A, artifacts=[{"path": str(tmp_path / "absent.pdf")}])
    assert store.count() == 0


def test_replace_keeps_reasoning_and_files_unless_given(store, tmp_path):
    src = tmp_path / "v1.pdf"
    src.write_bytes(b"v1")
    ep = store.add(Q, A, reasoning=R, artifacts=[{"path": str(src)}])

    kept = store.add(Q, "Réponse v2", replace_episode_id=ep["id"])
    assert kept["reasoning"] == R and [a["name"] for a in kept["artifacts"]] == ["v1.pdf"]
    assert kept["artifacts"][0]["path"]

    new_src = tmp_path / "v2.pdf"
    new_src.write_bytes(b"v2")
    changed = store.add(
        Q, "Réponse v3", reasoning="Nouvelle démarche.", artifacts=[{"path": str(new_src)}], replace_episode_id=ep["id"]
    )
    assert changed["reasoning"] == "Nouvelle démarche."
    assert [a["name"] for a in changed["artifacts"]] == ["v2.pdf"]
    assert sorted(p.name for p in (store.settings.artifacts_dir / ep["id"]).iterdir()) == ["v2.pdf"]
