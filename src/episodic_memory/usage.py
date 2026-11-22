"""Tokens d'un épisode : taille stockée (estimée) et consommation réelle de l'IA.

Deux mesures distinctes :

- **Taille** (`size_tokens`) : nombre de tokens estimé de la question, de la réponse et de la
  synthèse du raisonnement, c.-à-d. ce que coûte la *réutilisation* de l'épisode dans un
  contexte. Estimation à partir du nombre de caractères (pas de tokenizer Claude hors ligne).

- **Consommation** (`usage`) : tokens réellement consommés par l'IA pour produire la réponse
  (entrée, sortie, cache), tels que facturés par l'API. Deux sources :
    * `transcript` — lus dans le journal de la conversation Claude Code
      (`~/.claude/projects/<projet>/<session>.jsonl`, chaque réponse de l'API y porte son
      `usage`). La fenêtre va du message de l'utilisateur qui a lancé le travail (celui qui
      précède l'appel à `find_similar_question` le plus proche de la question stockée) jusqu'à
      l'appel à `store_validated_episode`. Les sous-agents de la session sont inclus.
    * `reported` — fournis par le client MCP lui-même (autres clients que Claude Code, ou
      serveur partagé qui ne voit pas le poste de l'utilisateur).
"""

from __future__ import annotations

import json
import math
import time
from collections.abc import Callable, Iterator
from pathlib import Path
from typing import Any

USAGE_FIELDS = ("input_tokens", "output_tokens", "cache_creation_input_tokens", "cache_read_input_tokens")
STORE_TOOL = "store_validated_episode"
SEARCH_TOOL = "find_similar_question"
# Journaux examinés pour retrouver l'appel de stockage en cours : modifiés récemment.
RECENT_SECONDS = 15 * 60


# --------------------------------------------------------------------- taille estimée


def estimate_tokens(text: str | None, chars_per_token: float = 4.0) -> int:
    """Estimation grossière (≈ 4 caractères par token pour du français/anglais)."""
    text = (text or "").strip()
    return math.ceil(len(text) / chars_per_token) if text else 0


def size_tokens(question: str, answer: str, reasoning: str | None, chars_per_token: float = 4.0) -> dict[str, int]:
    sizes = {
        "question": estimate_tokens(question, chars_per_token),
        "answer": estimate_tokens(answer, chars_per_token),
        "reasoning": estimate_tokens(reasoning, chars_per_token),
    }
    return sizes | {"total": sum(sizes.values())}


# ------------------------------------------------------------- consommation réelle


def normalize_reported(usage: dict[str, Any]) -> dict[str, Any]:
    """Consommation déclarée par le client : champs connus, entiers ≥ 0, total recalculé."""
    out: dict[str, Any] = {f: max(int(usage.get(f) or 0), 0) for f in USAGE_FIELDS}
    out["total_tokens"] = sum(out[f] for f in USAGE_FIELDS)
    out["fresh_tokens"] = out["input_tokens"] + out["cache_creation_input_tokens"] + out["output_tokens"]
    out["api_calls"] = max(int(usage.get("api_calls") or 0), 0) or None
    model = usage.get("model") or usage.get("models")
    out["models"] = [model] if isinstance(model, str) else list(model or [])
    out["source"] = "reported"
    return out


def _entries(path: Path) -> Iterator[dict[str, Any]]:
    try:
        with path.open(encoding="utf-8") as fh:
            for line in fh:
                try:
                    yield json.loads(line)
                except ValueError:
                    continue
    except OSError:
        return


def _tool_uses(entry: dict[str, Any]) -> Iterator[dict[str, Any]]:
    if entry.get("type") != "assistant":
        return
    content = (entry.get("message") or {}).get("content")
    for block in content if isinstance(content, list) else []:
        if isinstance(block, dict) and block.get("type") == "tool_use":
            yield block


def _is_tool(block: dict[str, Any], tool: str) -> bool:
    # Nom MCP complet : mcp__<serveur>__<outil>.
    return str(block.get("name", "")).split("__")[-1] == tool


def _is_human_prompt(entry: dict[str, Any]) -> bool:
    if entry.get("type") != "user" or entry.get("isMeta") or entry.get("isSidechain"):
        return False
    content = (entry.get("message") or {}).get("content")
    if isinstance(content, str):
        return bool(content.strip())
    # Un message composé de tool_result est un retour d'outil, pas un message de l'utilisateur.
    return isinstance(content, list) and any(
        isinstance(b, dict) and b.get("type") == "text" for b in content
    ) and not any(isinstance(b, dict) and b.get("type") == "tool_result" for b in content)


def find_transcript(projects_dir: Path, question: str, wait_seconds: float = 3.0) -> Path | None:
    """Journal Claude Code récent qui contient l'appel de stockage de cette question."""
    question = question.strip()
    deadline = time.monotonic() + wait_seconds
    while True:
        now = time.time()
        candidates = []
        for path in projects_dir.glob("*/*.jsonl"):
            try:
                mtime = path.stat().st_mtime
            except OSError:
                continue
            if now - mtime <= RECENT_SECONDS:
                candidates.append((mtime, path))
        for _, path in sorted(candidates, reverse=True):
            for entry in _entries(path):
                if any(
                    _is_tool(b, STORE_TOOL) and str((b.get("input") or {}).get("question", "")).strip() == question
                    for b in _tool_uses(entry)
                ):
                    return path
        if time.monotonic() >= deadline:
            return None
        time.sleep(0.5)  # l'appel en cours peut ne pas être encore écrit dans le journal


def transcript_usage(
    path: Path,
    question: str,
    similarity: Callable[[str, list[str]], list[float]] | None = None,
) -> dict[str, Any] | None:
    """Consommation de tokens pour la question stockée, lue dans un journal Claude Code.

    `similarity(question, candidats)` renvoie la similarité de la question stockée avec chaque
    question passée à `find_similar_question` ; elle sert à choisir le début de la fenêtre
    quand la conversation a traité plusieurs questions. Sans elle : le dernier appel.
    Renvoie None si l'appel de stockage n'est pas dans le journal.
    """
    question = question.strip()
    entries = [e for e in _entries(path) if e.get("type") in ("user", "assistant")]

    end = None  # dernier appel de stockage de CETTE question
    searches: list[tuple[int, str]] = []
    stores: list[tuple[int, str]] = []
    for i, entry in enumerate(entries):
        for block in _tool_uses(entry):
            args = block.get("input") or {}
            if _is_tool(block, STORE_TOOL):
                q = str(args.get("question", "")).strip()
                stores.append((i, q))
                if q == question:
                    end = i
            elif _is_tool(block, SEARCH_TOOL) and not entry.get("isSidechain"):
                searches.append((i, str(args.get("question", ""))))
    if end is None:
        return None
    # Le travail sur cette question commence après le stockage d'une AUTRE question.
    floor = max((i + 1 for i, q in stores if i < end and q != question), default=0)
    searches = [s for s in searches if floor <= s[0] < end]

    window = "question"
    if searches:
        if similarity is not None and len(searches) > 1:
            scores = similarity(question, [q for _, q in searches])
            best = max(scores)
            # À score égal, l'appel le plus récent.
            anchor = max(i for (i, _), s in zip(searches, scores) if s >= best - 1e-6)
        else:
            anchor = searches[-1][0]
        start = next((j for j in range(anchor, floor - 1, -1) if _is_human_prompt(entries[j])), floor)
    else:
        # Pas de recherche préalable : depuis le début de la conversation, ou depuis le stockage précédent.
        window = "session" if floor == 0 else "since_previous_store"
        start = next((j for j in range(floor, end + 1) if _is_human_prompt(entries[j])), floor)

    t_start = entries[start].get("timestamp", "")
    t_end = entries[end].get("timestamp", "")
    totals = {f: 0 for f in USAGE_FIELDS}
    seen: set[str] = set()
    models: set[str] = set()
    calls = sub_calls = 0

    def add(entry: dict[str, Any], sidechain: bool) -> None:
        nonlocal calls, sub_calls
        message = entry.get("message") or {}
        usage = message.get("usage")
        if not usage:
            return
        # Une réponse de l'API est écrite en plusieurs lignes (une par bloc), même usage : dédoublonner.
        key = message.get("id") or entry.get("requestId") or entry.get("uuid")
        if key in seen:
            return
        seen.add(key)
        for f in USAGE_FIELDS:
            totals[f] += int(usage.get(f) or 0)
        if message.get("model") and not str(message["model"]).startswith("<"):
            models.add(message["model"])
        calls += 1
        sub_calls += sidechain

    for entry in entries[start : end + 1]:
        if entry.get("type") == "assistant":
            add(entry, bool(entry.get("isSidechain")))
    # Sous-agents : journaux séparés dans <session>/subagents/.
    for sub in sorted(path.with_suffix("").glob("**/*.jsonl")):
        for entry in _entries(sub):
            if entry.get("type") == "assistant" and t_start <= entry.get("timestamp", "") <= t_end:
                add(entry, True)

    return totals | {
        "total_tokens": sum(totals.values()),
        # Hors relectures du cache (le même contexte relu à chaque appel) : ce qui a été réellement
        # traité pour la première fois ou produit.
        "fresh_tokens": totals["input_tokens"] + totals["cache_creation_input_tokens"] + totals["output_tokens"],
        "api_calls": calls,
        "subagent_calls": sub_calls,
        "models": sorted(models),
        "started_at": t_start,
        "ended_at": t_end,
        "window": window,
        "source": "transcript",
        "session_id": path.stem,
    }


def measure(
    projects_dir: Path,
    question: str,
    similarity: Callable[[str, list[str]], list[float]] | None = None,
    wait_seconds: float = 3.0,
) -> dict[str, Any] | None:
    """Consommation de la conversation en cours (Claude Code), ou None si introuvable."""
    if not projects_dir.is_dir():
        return None
    path = find_transcript(projects_dir, question, wait_seconds)
    return transcript_usage(path, question, similarity) if path else None


def scan_store_calls(projects_dir: Path) -> Iterator[tuple[Path, str]]:
    """Tous les appels de stockage passés (journal, question) : rattrapage des anciens épisodes."""
    for path in sorted(projects_dir.glob("*/*.jsonl")):
        questions: list[str] = []
        for entry in _entries(path):
            for block in _tool_uses(entry):
                if _is_tool(block, STORE_TOOL):
                    q = str((block.get("input") or {}).get("question", "")).strip()
                    if q and q not in questions:
                        questions.append(q)
        for q in questions:
            yield path, q
