"""Hook Claude Code `UserPromptSubmit` : rappelle le protocole de mémoire épisodique.

Les instructions d'un serveur MCP ne sont qu'indicatives : l'assistant peut les ignorer.
Ce hook est exécuté par Claude Code lui-même à chaque message de l'utilisateur et injecte
un court rappel dans le contexte, pour que la recherche d'une question proche (avant la
réponse) et la proposition de stockage (après) ne dépendent plus de sa seule initiative.

Volontairement léger : aucune dépendance, pas de chargement de la base ni du modèle.

    python -m episodic_memory.hook      (lit le JSON du hook sur stdin)

Le même module sert de hook `PostToolUse` sur `store_validated_episode` : il mesure dans le
journal de la conversation (sur le poste de l'utilisateur) les tokens consommés pour la
question stockée, et demande à l'assistant de les enregistrer avec `set_episode_usage`.
Indispensable avec un serveur partagé (http), qui ne voit pas ce journal.

Désactivation ponctuelle : variable d'environnement EPISODIC_HOOK_DISABLED=1.
"""

from __future__ import annotations

import json
import os
import sys

REMINDER = (
    "[Mémoire épisodique — rappel automatique du hook, pas un message de l'utilisateur] "
    "Si ce message est une question ou une demande de travail (analyse, rapport, fichier à produire, "
    "explication, diagnostic, développement) : "
    "1) AVANT de répondre, appelle l'outil `find_similar_question` du serveur MCP `episodic-memory` "
    "avec la demande telle que posée (si l'outil n'est pas encore chargé, charge-le d'abord). "
    "Si une question proche est trouvée, propose-la à l'utilisateur ; s'il confirme, appelle "
    "`get_validated_episode` et appuie-toi sur la réponse, le raisonnement et les fichiers stockés. "
    "2) APRÈS ta réponse finale, propose explicitement à l'utilisateur d'enregistrer l'épisode "
    "(question, réponse, synthèse du raisonnement, fichiers produits) en lui montrant la synthèse ; "
    "n'appelle `store_validated_episode` qu'après son accord explicite. "
    "Ignore ce rappel si le message est une simple confirmation, un remerciement, ou la réponse à "
    "une question que tu viens de poser — sauf s'il valide justement un stockage ou une question proche."
)

MIN_WORDS = 4


def reminder_for(prompt: str) -> str | None:
    """Le rappel à injecter pour ce message, ou None s'il ne le justifie pas."""
    text = prompt.strip()
    if not text or text.startswith("/"):  # commande slash : pas une question
        return None
    if len(text.split()) < MIN_WORDS:  # « oui », « ok merci », « vas-y »…
        return None
    return REMINDER


def _stored_episode(response: object) -> dict | None:
    """L'épisode renvoyé par `store_validated_episode` (réponse MCP brute ou déjà décodée)."""
    if isinstance(response, dict) and "status" in response:
        data = response
    else:
        texts = []
        blocks = response if isinstance(response, list) else (response.get("content") if isinstance(response, dict) else None)
        for block in blocks or []:
            if isinstance(block, dict) and isinstance(block.get("text"), str):
                texts.append(block["text"])
        if isinstance(response, str):
            texts.append(response)
        data = None
        for text in texts:
            try:
                data = json.loads(text)
                break
            except ValueError:
                continue
        if not isinstance(data, dict):
            return None
        data = data.get("result", data) if "status" not in data else data
    if data.get("status") not in ("stored", "replaced"):
        return None
    episode = data.get("episode")
    return episode if isinstance(episode, dict) and episode.get("id") else None


def usage_context(payload: dict) -> str | None:
    """Consigne `PostToolUse` : enregistrer la consommation mesurée de l'épisode stocké."""
    if not str(payload.get("tool_name", "")).endswith("__store_validated_episode"):
        return None
    episode = _stored_episode(payload.get("tool_response"))
    if episode is None or episode.get("usage"):  # échec, ou déjà mesuré par le serveur (stdio)
        return None
    question = str((payload.get("tool_input") or {}).get("question", ""))
    transcript = payload.get("transcript_path")
    if not question or not transcript:
        return None
    from pathlib import Path

    from .usage import transcript_usage

    usage = transcript_usage(Path(transcript), question)
    if usage is None:
        return None
    keys = (
        "input_tokens", "output_tokens", "cache_creation_input_tokens", "cache_read_input_tokens",
        "api_calls", "subagent_calls", "window", "started_at", "ended_at",
    )
    args = {k: usage[k] for k in keys} | {"model": ", ".join(usage["models"]) or None}
    return (
        "[Mémoire épisodique — hook, pas un message de l'utilisateur] Tokens consommés pour l'épisode "
        f"{episode['id']}, mesurés dans le journal de cette conversation : {usage['total_tokens']} au total "
        f"(dont {usage['output_tokens']} en sortie et {usage['cache_read_input_tokens']} relus du cache, "
        f"{usage['api_calls']} appels au modèle). Appelle maintenant l'outil `set_episode_usage` du serveur "
        f"`episodic-memory` avec episode_id=\"{episode['id']}\" et usage={json.dumps(args)} (recopie exacte, "
        "sans redemander l'accord de l'utilisateur), puis indique-lui ces chiffres en une ligne."
    )


def main() -> None:
    if os.environ.get("EPISODIC_HOOK_DISABLED") == "1":
        return
    try:
        payload = json.loads(sys.stdin.buffer.read().decode("utf-8") or "{}")
    except (ValueError, UnicodeDecodeError):
        payload = {}
    if payload.get("hook_event_name") == "PostToolUse":
        try:
            context = usage_context(payload)
        except Exception:  # un hook ne doit jamais bloquer l'assistant
            context = None
        if context:
            print(json.dumps({"hookSpecificOutput": {"hookEventName": "PostToolUse", "additionalContext": context}}))
        return
    reminder = reminder_for(str(payload.get("prompt", "")))
    if reminder:
        # json.dumps échappe les accents en ASCII : aucun souci d'encodage de console sous Windows.
        print(json.dumps({"hookSpecificOutput": {"hookEventName": "UserPromptSubmit", "additionalContext": reminder}}))


if __name__ == "__main__":
    main()
